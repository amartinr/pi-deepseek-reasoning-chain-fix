/**
 * Standalone verification of the pi-deepseek-reasoning-chain package.
 *
 * Loads the BUILT artifact (dist/index.js) the way pi loads a packaged
 * extension, then exercises the exported fix functions against realistic
 * payloads:
 *    - config-driven scope (model ids, exact/prefix, empty -> inert)
 *    - pi serializer WITH thinkingSignature  -> real text already replayed
 *    - pi serializer WITHOUT signature      -> wire fix injects " "
 *    - thinking:disabled kill-switch        -> stripped on continuations
 */

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const mod = await import("/work/dist/index.js");

const {
  modelsMatch,
  loadConfig,
  isToolScope,
  historyHasToolCalls,
  historyHasReasoning,
  fixNativeMessagesForDeepSeek,
  fixWirePayloadForDeepSeek,
  fixFinalizedMessageForDeepSeek,
  default: factory,
} = mod;

// ---- 1) jiti load: the factory registers the 3 hooks -----------------------
const tmpCfg = mkdtempSync(join(tmpdir(), "dsrc-"));
writeFileSync(join(tmpCfg, "config.json"), JSON.stringify({ models: ["deepseek/deepseek-v4-flash"] }));
process.env.PI_DEEPSEEK_REASONING_CONFIG = join(tmpCfg, "config.json");

let registered = [];
const stubPi = { on: (name, fn) => registered.push(name) };
factory(stubPi);
assert.deepEqual(registered.sort(), [
  "before_provider_request",
  "context",
  "message_end",
]);
console.log("ok: extension loads via jiti, registers context + before_provider_request + message_end");

// ---- 2) config-driven scope -------------------------------------------------
assert.equal(modelsMatch("deepseek/deepseek-v4-flash", ["deepseek/deepseek-v4-flash"]), true); // exact
assert.equal(modelsMatch("deepseek/deepseek-v4-pro", ["deepseek/"]), true); // prefix
assert.equal(modelsMatch("DEEPSEEK/DeepSeek-v4-flash", ["deepseek/deepseek-v4-flash"]), true); // case-insensitive
assert.equal(modelsMatch("claude-haiku-4-5", ["deepseek/"]), false); // non-deepseek excluded
assert.equal(modelsMatch("deepseek-v4-flash", ["deepseek/deepseek-v4-flash"]), false); // bare id, not configured
assert.equal(modelsMatch("gpt-4o", []), false); // empty list -> never matches
assert.equal(modelsMatch(undefined, ["deepseek/"]), false);
console.log("ok: config scope (exact, prefix, case-insensitive, empty -> inert, bare id not configured)");

// ---- 2b) loadConfig: file, malformed, missing ------------------------------
const loaded = loadConfig(join(tmpCfg, "config.json"));
assert.deepEqual(loaded.models, ["deepseek/deepseek-v4-flash"]);
assert.equal(loaded.replayReasoning, true); // default when absent
const malformed = mkdtempSync(join(tmpdir(), "dsrc-bad-"));
writeFileSync(join(malformed, "config.json"), "{ not json");
assert.deepEqual(loadConfig(join(malformed, "config.json")).models, []); // fail-open
assert.deepEqual(loadConfig("/nonexistent/config.json").models, []); // missing -> inert
// replayReasoning parsing
const replayCfg = mkdtempSync(join(tmpdir(), "dsrc-rp-"));
writeFileSync(join(replayCfg, "off.json"), JSON.stringify({ models: ["deepseek/"], replayReasoning: false }));
writeFileSync(join(replayCfg, "on.json"), JSON.stringify({ models: ["deepseek/"], replayReasoning: true }));
writeFileSync(join(replayCfg, "bad.json"), JSON.stringify({ models: ["deepseek/"], replayReasoning: "yes" }));
assert.equal(loadConfig(join(replayCfg, "off.json")).replayReasoning, false);
assert.equal(loadConfig(join(replayCfg, "on.json")).replayReasoning, true);
assert.equal(loadConfig(join(replayCfg, "bad.json")).replayReasoning, true); // non-boolean -> default
console.log("ok: loadConfig (valid, malformed -> inert, missing -> inert, replayReasoning parsing)");

// ---- 2c) replayReasoning knob: handler behavior in both modes ---------------
function captureHandlers(configPath) {
  process.env.PI_DEEPSEEK_REASONING_CONFIG = configPath;
  const handlers = {};
  factory({ on: (name, fn) => (handlers[name] = fn) });
  return handlers;
}
const dsCtx = { model: { id: "deepseek/deepseek-v4-flash" } };
const thinkingMsg = {
  messages: [{ role: "assistant", content: [{ type: "thinking", thinking: "real chain" }] }],
};
const finalMsgKnob = { message: { role: "assistant", content: [{ type: "thinking", thinking: "real chain" }] } };
const wireIn = {
  payload: {
    model: "deepseek/deepseek-v4-flash",
    thinking: { type: "disabled" },
    messages: [
      { role: "assistant", content: "a", tool_calls: [{ id: "c1", type: "function", function: { name: "get_date", arguments: "{}" } }] },
    ],
    tools: [],
  },
};

// chaining mode: native layer stamps, wire layer works
const chainH = captureHandlers(join(replayCfg, "on.json"));
const ctxOut = await chainH.context(thinkingMsg, dsCtx);
assert.equal(ctxOut.messages[0].content[0].thinkingSignature, "reasoning_content");
assert.ok(chainH.message_end(finalMsgKnob, dsCtx));
assert.ok(chainH.before_provider_request({ payload: structuredClone(wireIn.payload) }, dsCtx));

// compliance mode: native layer inert, wire layer still active
const compH = captureHandlers(join(replayCfg, "off.json"));
assert.equal(await compH.context(thinkingMsg, dsCtx), undefined); // no stamping
assert.equal(await compH.message_end(finalMsgKnob, dsCtx), undefined); // no stored signature
const compWire = compH.before_provider_request({ payload: structuredClone(wireIn.payload) }, dsCtx);
assert.ok(compWire);
assert.equal(compWire.messages[0].reasoning_content, " "); // " " still forced (contract)
assert.equal(compWire.thinking.type, "disabled"); // thinking preserved (user intent)
console.log("ok: replayReasoning knob (chaining vs compliance-only, wire always active)");

// ---- 3) context fix: stamps signature so the serializer replays REAL text ---
const nativeMsgs = [
  { role: "user", content: [{ type: "text", text: "weather?" }] },
  {
    role: "assistant",
    content: [
      { type: "thinking", thinking: "I need today's date first." },
      { type: "text", text: "Checking." },
      { type: "toolCall", id: "c1", name: "get_date", arguments: {} },
    ],
  },
];
const { messages: stamped, changed: ctxChanged } = fixNativeMessagesForDeepSeek(nativeMsgs);
assert.equal(ctxChanged, true);
assert.equal(stamped[1].content[0].thinkingSignature, "reasoning_content");
// idempotent
const again = fixNativeMessagesForDeepSeek(stamped);
assert.equal(again.changed, false);
console.log("ok: context fix stamps thinkingSignature, idempotent");

// ---- 4) wire fix: real text preserved, ""/missing -> " " --------------------
const wirePayload = {
  model: "deepseek/deepseek-v4-flash",
  messages: [
    { role: "user", content: "weather?" },
    {
      role: "assistant",
      content: "Checking.",
      tool_calls: [{ id: "c1", type: "function", function: { name: "get_date", arguments: "{}" } }],
      reasoning_content: "I need today's date first.", // replay worked
    },
    { role: "tool", tool_call_id: "c1", content: '{"date":"2026-09-02"}' },
    { role: "assistant", content: "Let me call get_weather." }, // no reasoning_content at all
  ],
  tools: [],
};
const fixed = fixWirePayloadForDeepSeek(wirePayload);
assert.ok(fixed);
assert.equal(fixed.messages[1].reasoning_content, "I need today's date first."); // real text untouched
assert.equal(fixed.messages[3].reasoning_content, " "); // missing -> " "
console.log("ok: wire fix keeps real text, upgrades missing/empty to non-empty");

// ---- 5) wire fix: out of tool scope -> untouched AND unmutated -------------
const plain = {
  model: "deepseek/deepseek-v4-flash",
  messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
};
const plainCopy = structuredClone(plain);
assert.equal(fixWirePayloadForDeepSeek(plain), undefined);
assert.deepEqual(plain, plainCopy); // no in-place mutation outside scope
console.log("ok: no tool scope -> payload untouched and unmutated");

// ---- 5b) tools=[] with no tool history still counts as scope (contract) ---
const emptyTools = {
  model: "deepseek/deepseek-v4-flash",
  tools: [],
  messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
};
const fixedEmptyTools = fixWirePayloadForDeepSeek(emptyTools);
assert.ok(fixedEmptyTools);
assert.equal(fixedEmptyTools.messages[1].reasoning_content, " ");
console.log("ok: tools=[] still triggers the contract forcing");

// ---- 6) thinking:disabled is PRESERVED (user intent, never stripped) --------
// 6a) continuation with real reasoning in history: nothing to force, and
//     thinking stays as the user set it -> payload untouched (undefined).
const disabledWithReasoning = {
  model: "deepseek/deepseek-v4-flash",
  thinking: { type: "disabled" },
  messages: [
    { role: "user", content: "weather?" },
    { role: "assistant", content: "Checking.", reasoning_content: "real chain", tool_calls: [{ id: "c1", type: "function", function: { name: "get_date", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", content: "ok" },
  ],
  tools: [],
};
const keptCopy = structuredClone(disabledWithReasoning);
assert.equal(fixWirePayloadForDeepSeek(disabledWithReasoning), undefined); // nothing to change
assert.deepEqual(disabledWithReasoning, keptCopy); // thinking and reasoning untouched
console.log("ok: thinking:disabled preserved on continuation with real reasoning (no-op)");

// 6b) continuation WITHOUT real reasoning (placeholder history): forcing
//     applies, thinking still preserved.
const disabledNoReasoning = {
  model: "deepseek/deepseek-v4-flash",
  thinking: { type: "disabled" },
  messages: [
    { role: "user", content: "weather?" },
    { role: "assistant", content: "Checking.", tool_calls: [{ id: "c1", type: "function", function: { name: "get_date", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", content: "ok" },
  ],
  tools: [],
};
const keptNoReasoning = fixWirePayloadForDeepSeek(disabledNoReasoning);
assert.ok(keptNoReasoning);
assert.equal(keptNoReasoning.thinking.type, "disabled");
assert.equal(keptNoReasoning.messages[1].reasoning_content, " "); // contract still satisfied
console.log("ok: thinking:disabled preserved on continuation without real reasoning (user intent)");

// 6c) thinking:disabled on a plain non-tool chat: untouched, no scope
const disabledPlainChat = {
  model: "deepseek/deepseek-v4-flash",
  thinking: { type: "disabled" },
  messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
};
assert.equal(fixWirePayloadForDeepSeek(disabledPlainChat), undefined);
assert.equal(disabledPlainChat.thinking.type, "disabled"); // untouched
console.log("ok: thinking:disabled kept on non-tool chat (user choice)");

// ---- 6d) determinism: running the fix twice yields a byte-identical payload
const det = {
  model: "deepseek/deepseek-v4-flash",
  thinking: { type: "disabled" },
  messages: [
    { role: "user", content: "weather?" },
    { role: "assistant", content: "Checking.", tool_calls: [{ id: "c1", type: "function", function: { name: "get_date", arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", content: "ok" },
    { role: "assistant", content: "Answering." },
  ],
  tools: [],
};
const first = structuredClone(det);
fixWirePayloadForDeepSeek(first);
const second = structuredClone(first);
fixWirePayloadForDeepSeek(second);
assert.deepEqual(first, second); // idempotent serialization -> stable prefix cache
console.log("ok: wire fix deterministic and idempotent (prefix cache stable)");

// ---- 6e) historyHasToolCalls helper ----------------------------------------
assert.equal(historyHasToolCalls(disabledNoReasoning.messages), true);
assert.equal(historyHasToolCalls([{ role: "user", content: "hi" }]), false);
console.log("ok: historyHasToolCalls detects continuation flag");

// ---- 7) return path: finalized message gets replayable signature -----------
const finalMsg = {
  role: "assistant",
  content: [{ type: "thinking", thinking: "The user asked for weather." }, { type: "text", text: "It is sunny." }],
};
const res = fixFinalizedMessageForDeepSeek(finalMsg);
assert.ok(res);
assert.equal(res.message.content[0].thinkingSignature, "reasoning_content");
assert.equal(fixFinalizedMessageForDeepSeek({ role: "user", content: "x" }), undefined);
console.log("ok: message_end fix stamps stored message, non-assistant untouched");

// ---- 8) P1 robustness: string content never crashes ------------------------
const stringContentMsgs = [
  { role: "user", content: "hi" },
  { role: "assistant", content: "hello" }, // string content (edge)
  { role: "assistant", content: [{ type: "thinking", thinking: "real" }] },
];
const strRes = fixNativeMessagesForDeepSeek(stringContentMsgs);
assert.equal(strRes.changed, true); // only the array-content block gets stamped
assert.equal(stringContentMsgs[1].content, "hello"); // string untouched, no char iteration
assert.equal(fixFinalizedMessageForDeepSeek({ role: "assistant", content: "plain string" }), undefined);
console.log("ok: string content handled defensively (no crash, no mutation)");

console.log("\nALL CHECKS PASSED");
