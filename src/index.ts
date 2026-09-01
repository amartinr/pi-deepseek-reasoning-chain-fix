/**
 * DeepSeek Reasoning Chain — pi extension
 *
 * Fixes the DeepSeek reasoning-chaining contract on tool-call turns.
 *
 * ## The contract (DeepSeek API docs, Thinking Mode → Tool Calls)
 *
 * "For requests carrying the `tools` parameter, the `reasoning_content` must
 * be fully passed back to the API in all subsequent requests — even for
 * turns where the model did not perform a tool call. If your code does not
 * correctly pass back `reasoning_content`, the API will return a 400 error."
 *
 * When `tools` is in scope the model's chain-of-thought IS concatenated into
 * the context, so each continuation can continue the previous reasoning.
 * A missing/empty field means a blank chain (silent degradation through
 * gateways like LiteLLM, which inject their own placeholder and warn), and
 * `thinking: {"type": "disabled"}` on a continuation is a hard kill-switch
 * (0 reasoning deltas, verified live).
 *
 * ## What pi already does (openai-completions serializer)
 *
 * - Replays the REAL reasoning text as `reasoning_content` only when the
 *   stored `thinking` content block carries a recognized `thinkingSignature`
 *   (set at stream time). Blocks without a signature fall back to nothing.
 * - Forces `reasoning_content = ""` on assistant messages, but ONLY when
 *   pi's own DeepSeek detection fires (provider === "deepseek" or baseUrl
 *   contains "deepseek.com"). Behind a gateway (e.g. LiteLLM at
 *   litellm.private) that detection never fires.
 * - Sends `thinking: {"type": "disabled"}` when reasoning is off, which
 *   kills reasoning on tool-call continuations.
 *
 * ## What this extension does (3 hooks)
 *
 * 1. `context` — native message layer, before serialization: stamp every
 *    non-empty thinking block with `thinkingSignature = "reasoning_content"`
 *    so pi's serializer replays the REAL text on continuations (idempotent;
 *    covers blocks whose signature was lost, e.g. resumed sessions).
 * 2. `before_provider_request` — wire payload layer, the last hook before
 *    the request reaches the endpoint: in tool scope, force a non-empty
 *    `reasoning_content` (" ") on every assistant message. `thinking` is
 *    deliberately left alone — pi sends `thinking: disabled` only when the
 *    user chose thinking off, and stripping it would override that choice.
 * 3. `message_end` — return path: normalize the finalized assistant message
 *    so the STORED reasoning stays replayable by the next continuation
 *    (ensures `thinkingSignature` survives whatever the session persistence
 *    does with extra block fields).
 *
 * Scope is explicit, not heuristic: the model ids the fix applies to are
 * listed in a config file (see the Configuration section below). All changes
 * are deterministic and fail open.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import os from "node:os";

// ----------------------------------------------------------------------------
// Configuration (model ids the extension applies to)
// ----------------------------------------------------------------------------
// The scope is explicit, not heuristic: the config file lists the model ids
// for which the DeepSeek contract fix applies. Keeping it in a file (instead
// of provider/baseUrl sniffing) avoids silent misses and false positives.
//
//   ~/.pi/agent/extensions/pi-deepseek-reasoning-chain-fix/config.json
//   { "models": ["deepseek/deepseek-v4-flash", "deepseek/"] }
//
// An id matches when the model id equals the entry or starts with it (case
// insensitive), so "deepseek/" covers every deepseek-routed model. An empty
// or missing list leaves the extension inert — the safe default.
//
// Override the path with PI_DEEPSEEK_REASONING_CONFIG (tests, custom setups).

export interface ExtensionConfig {
  models: string[];
  /** true = real reasoning text replayed on continuations (chaining);
   *  false = compliance only, " " placeholder sent (no chaining). Default true. */
  replayReasoning: boolean;
}

const DEFAULT_CONFIG_PATH = join(
  os.homedir(),
  ".pi",
  "agent",
  "extensions",
  "pi-deepseek-reasoning-chain-fix",
  "config.json"
);

/** Load the extension config. Fail-open: any error yields an empty (inert) list. */
export function loadConfig(
  configPath: string = process.env.PI_DEEPSEEK_REASONING_CONFIG ?? DEFAULT_CONFIG_PATH
): ExtensionConfig {
  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf8"));
    const models = Array.isArray(parsed?.models)
      ? parsed.models
          .filter((m: unknown): m is string => typeof m === "string" && m.trim().length > 0)
          .map((m: string) => m.trim().toLowerCase())
      : [];
    const replayReasoning = typeof parsed?.replayReasoning !== "boolean" ? true : parsed.replayReasoning;
    return { models, replayReasoning };
  } catch {
    return { models: [], replayReasoning: true }; // missing/malformed file -> inert
  }
}

/** True when the model id equals a configured id or starts with one. */
export function modelsMatch(modelId: string | undefined, models: string[]): boolean {
  if (!modelId || models.length === 0) return false;
  const id = modelId.toLowerCase();
  return models.some((m) => id === m || id.startsWith(m));
}

export function isToolScope(payload: Record<string, any>): boolean {
  if (payload.tools !== undefined) return true; // [] still carries the tools parameter
  return (payload.messages ?? []).some(
    (m: any) => m?.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length > 0
  );
}

/** True when the history contains a prior assistant that produced a tool call. */
export function historyHasToolCalls(messages: any[]): boolean {
  return messages.some(
    (m) => m?.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length > 0
  );
}

/** Does the history contain any real reasoning to chain? */
export function historyHasReasoning(messages: any[]): boolean {
  return messages.some(
    (m) => m?.role === "assistant" && typeof m.reasoning_content === "string" && m.reasoning_content.trim().length > 0
  );
}

/** 1) context fix: stamp thinking blocks so the serializer replays REAL text. */
export function fixNativeMessagesForDeepSeek(messages: any[]): { messages: any[]; changed: boolean } {
  let changed = false;
  for (const msg of messages) {
    if (msg.role !== "assistant") continue;
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking.trim()) {
        if (block.thinkingSignature !== "reasoning_content") {
          block.thinkingSignature = "reasoning_content";
          changed = true;
        }
      }
    }
  }
  return { messages, changed };
}

/**
 * 2) wire fix: force non-empty reasoning_content (DeepSeek contract).
 *
 * Single pass over messages computes the tool-scope flag and the list of
 * assistant messages needing the " " placeholder — mutations are applied
 * only after the scope decision, so an out-of-scope payload is never
 * touched.
 *
 * Deliberately does NOT touch `thinking`: pi sends thinking:disabled only
 * when the user chose thinking off — stripping it would override that
 * choice. (The Open WebUI pipe stripped it because Open WebUI injects the
 * marker on continuations regardless of user intent; that premise does not
 * hold in pi.)
 */
export function fixWirePayloadForDeepSeek(payload: Record<string, any>): Record<string, any> | undefined {
  const messages = payload?.messages;
  if (!payload || !Array.isArray(messages)) return undefined;

  const hasToolsParam = payload.tools !== undefined; // [] still carries the tools parameter
  let hasAssistantToolCalls = false;
  const toForce: number[] = []; // assistant indices needing a non-empty placeholder

  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (!m || m.role !== "assistant") continue;
    const tcs = m.tool_calls;
    if (Array.isArray(tcs) && tcs.length > 0) hasAssistantToolCalls = true;
    const rc = m.reasoning_content;
    if (typeof rc !== "string" || rc.length === 0) toForce.push(i);
  }

  // Tool scope: the request carries `tools` (even []) or the history has a
  // prior assistant that produced a tool call (the DeepSeek contract is
  // driven by history content, not by this request's `tools`).
  if (!hasToolsParam && !hasAssistantToolCalls) return undefined;

  let changed = false;
  for (const i of toForce) {
    messages[i].reasoning_content = " ";
    changed = true;
  }

  return changed ? payload : undefined;
}

/** 3) return-path fix: keep the finalized assistant message replayable. */
export function fixFinalizedMessageForDeepSeek(message: any): { message: any } | undefined {
  if (message?.role !== "assistant") return undefined;
  let changed = false;
  if (!Array.isArray(message.content)) return undefined;
  for (const block of message.content) {
    if (block.type === "thinking" && typeof block.thinking === "string" && block.thinking.trim()) {
      if (block.thinkingSignature !== "reasoning_content") {
        block.thinkingSignature = "reasoning_content";
        changed = true;
      }
    }
  }
  return changed ? { message } : undefined;
}

export default function (pi: ExtensionAPI) {
  const config = loadConfig();
  const applies = (model: any): boolean => modelsMatch(model?.id, config.models);
  if (config.models.length === 0) {
    console.log(
      "[deepseek-reasoning-chain] loaded but INERT: no model ids configured " +
        "(set ~/.pi/agent/extensions/pi-deepseek-reasoning-chain-fix/config.json)"
    );
  } else {
    console.log(
      `[deepseek-reasoning-chain] active for ${config.models.length} model id(s): ${config.models.join(", ")}` +
        (config.replayReasoning ? " (reasoning replay on)" : " (compliance only: ' ' placeholder)")
    );
  }

  pi.on("context", async (event, ctx) => {
    // Native layer = chaining. In compliance-only mode the real reasoning
    // text is intentionally NOT replayed (and not stamped on storage), so
    // only the wire contract is satisfied.
    if (!config.replayReasoning) return;
    if (!applies(ctx.model)) return;
    const { messages, changed } = fixNativeMessagesForDeepSeek(event.messages);
    return changed ? { messages } : undefined;
  });

  pi.on("before_provider_request", (event, ctx) => {
    // Wire layer = contract compliance; active in both modes.
    if (!applies(ctx.model)) return;
    return fixWirePayloadForDeepSeek(event.payload as Record<string, any>);
  });

  pi.on("message_end", async (event, ctx) => {
    if (!config.replayReasoning) return; // compliance mode: keep stored history replay-free
    if (!applies(ctx.model)) return;
    return fixFinalizedMessageForDeepSeek(event.message);
  });
}
