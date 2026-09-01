/**
 * Standalone verification of the pi-deepseek-reasoning-chain package.
 *
 * Loads the BUILT artifact (dist/index.js) the way pi loads a packaged
 * extension, then exercises the exported fix functions against realistic
 * payloads:
 *    - pi serializer WITH thinkingSignature  -> real text already replayed
 *    - pi serializer WITHOUT signature      -> wire fix injects " "
 *    - LiteLLM gateway (model deepseek/deepseek-v4-flash, no pi deepseek
 *      detection)                           -> scope detection works
 *    - thinking:disabled kill-switch        -> stripped when history reasons
 */

import assert from "node:assert/strict";

const mod = await import("/work/dist/index.js");

const {
  isDeepSeekModel,
  isToolScope,
  historyHasReasoning,
  fixNativeMessagesForDeepSeek,
  fixWirePayloadForDeepSeek,
  fixFinalizedMessageForDeepSeek,
  default: factory,
} = mod;

// ---- 1) jiti load: the factory registers the 3 hooks -----------------------
let registered = [];
const stubPi = { on: (name, fn) => registered.push(name) };
factory(stubPi);
assert.deepEqual(registered.sort(), [
  "before_provider_request",
  "context",
  "message_end",
]);
console.log("ok: extension loads via jiti, registers context + before_provider_request + message_end");

// ---- 2) scope detection -----------------------------------------------------
assert.equal(isDeepSeekModel("deepseek", "deepseek-v4-flash", undefined), true);
assert.equal(isDeepSeekModel("litellm", "deepseek/deepseek-v4-flash", "http://litellm.private"), true);
assert.equal(isDeepSeekModel("openai", "deepseek-v4-pro", "http://litellm.private"), true);
assert.equal(isDeepSeekModel("openai", "deepseek/deepseek-v4-pro", "https://api.deepseek.com/v1"), true);
assert.equal(isDeepSeekModel("openai", "gpt-4o", "https://api.openai.com"), false);
assert.equal(isDeepSeekModel("anthropic", "claude-haiku-4-5", "http://litellm.private"), false);
console.log("ok: scope detection (direct, LiteLLM prefix, baseUrl; non-deepseek excluded)");

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

// ---- 5) wire fix: out of tool scope -> untouched ---------------------------
const plain = {
  model: "deepseek/deepseek-v4-flash",
  messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "hello" }],
};
assert.equal(fixWirePayloadForDeepSeek(plain), undefined);
console.log("ok: no tool scope -> payload untouched");

// ---- 6) thinking:disabled stripped only when history reasons ---------------
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
const stripped = fixWirePayloadForDeepSeek(disabledWithReasoning);
assert.ok(stripped);
assert.equal("thinking" in stripped, false);
console.log("ok: thinking:disabled stripped when history has reasoning");

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

console.log("\nALL CHECKS PASSED");
