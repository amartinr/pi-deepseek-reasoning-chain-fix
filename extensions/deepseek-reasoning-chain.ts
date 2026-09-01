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
 *    `reasoning_content` (" ") on every assistant message and strip
 *    `thinking: disabled` when the conversation has been reasoning.
 * 3. `message_end` — return path: normalize the finalized assistant message
 *    so the STORED reasoning stays replayable by the next continuation
 *    (ensures `thinkingSignature` survives whatever the session persistence
 *    does with extra block fields).
 *
 * Scope detection covers direct DeepSeek, DeepSeek behind any gateway that
 * keeps the `deepseek/` model prefix (LiteLLM), and any baseUrl containing
 * "deepseek.com". All changes are deterministic and fail open.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/** Extra model ids/prefixes to treat as DeepSeek (comma-separated env override). */
const EXTRA_PATTERN =
  process.env.PI_DEEPSEEK_REASONING_EXTRA?.split(",")
    .map((s) => s.trim())
    .filter(Boolean) ?? [];

export function isDeepSeekModel(
  provider: string | undefined,
  modelId: string | undefined,
  baseUrl: string | undefined
): boolean {
  if (!modelId) return false;
  if (provider === "deepseek") return true;
  if ((baseUrl ?? "").toLowerCase().includes("deepseek.com")) return true;
  if (modelId.startsWith("deepseek/") || modelId.startsWith("deepseek-")) return true;
  return EXTRA_PATTERN.some((p) => modelId.startsWith(p) || modelId === p);
}

export function isToolScope(payload: Record<string, any>): boolean {
  if (payload.tools !== undefined) return true; // [] still carries the tools parameter
  return (payload.messages ?? []).some(
    (m: any) => m?.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls.length > 0
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
    for (const block of msg.content ?? []) {
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

/** 2) wire fix: force non-empty reasoning_content + strip thinking:disabled. */
export function fixWirePayloadForDeepSeek(payload: Record<string, any>): Record<string, any> | undefined {
  const messages = payload?.messages;
  if (!payload || !Array.isArray(messages) || !isToolScope(payload)) return undefined;

  let changed = false;
  for (const m of messages) {
    if (m?.role !== "assistant") continue;
    const rc = m.reasoning_content;
    if (typeof rc !== "string" || rc.length === 0) {
      m.reasoning_content = " ";
      changed = true;
    }
  }

  const thinking = payload.thinking;
  if (thinking && typeof thinking === "object" && thinking.type === "disabled") {
    if (historyHasReasoning(messages)) {
      delete payload.thinking;
      changed = true;
    }
  }

  return changed ? payload : undefined;
}

/** 3) return-path fix: keep the finalized assistant message replayable. */
export function fixFinalizedMessageForDeepSeek(message: any): { message: any } | undefined {
  if (message?.role !== "assistant") return undefined;
  let changed = false;
  for (const block of message.content ?? []) {
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
  pi.on("context", async (event, ctx) => {
    const model = ctx.model as any;
    if (!model || !isDeepSeekModel(model.provider, model.id, model.baseUrl)) return;
    const { messages, changed } = fixNativeMessagesForDeepSeek(event.messages);
    return changed ? { messages } : undefined;
  });

  pi.on("before_provider_request", (event, ctx) => {
    const model = ctx.model as any;
    if (!model || !isDeepSeekModel(model.provider, model.id, model.baseUrl)) return;
    return fixWirePayloadForDeepSeek(event.payload as Record<string, any>);
  });

  pi.on("message_end", async (event, ctx) => {
    const model = ctx.model as any;
    if (!model || !isDeepSeekModel(model.provider, model.id, model.baseUrl)) return;
    return fixFinalizedMessageForDeepSeek(event.message);
  });
}
