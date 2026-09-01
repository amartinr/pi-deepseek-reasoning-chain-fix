/**
 * Live A/B of the pi-deepseek-reasoning-chain package against
 * http://litellm.private (deepseek/deepseek-v4-flash).
 *
 * Loads the BUILT artifact (dist/index.js) and simulates pi's full flow on
 * a tool-call continuation:
 *
 *   turn 1 (authentic) -> native stored message -> pi serializer -> wire
 *   payload -> [extension fixes] -> gateway
 *
 * Three conditions (the pipe's A/B):
 *   A) no extension      : assistant rebuilt WITHOUT reasoning_content
 *   B) wire fix only     : fixWirePayloadForDeepSeek injects " "
 *   C) native + wire fix : fixNativeMessagesForDeepSeek stamps the signature
 *                          so the serializer replays the REAL text
 *
 * N repetitions per condition; median reasoning length reported.
 */

import { readFileSync } from "node:fs";

const { fixNativeMessagesForDeepSeek, fixWirePayloadForDeepSeek } = await import(
  "/work/dist/index.js"
);

const KEY = readFileSync("/tmp/.litellm_key", "utf8").trim();
const URL = "http://litellm.private/v1/chat/completions";
const MODEL = "deepseek/deepseek-v4-flash";
const HEADERS = { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" };

const TOOLS = [
  { type: "function", function: { name: "get_date", description: "Get today's date.", parameters: { type: "object", properties: {} } } },
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get weather of a city.",
      parameters: { type: "object", properties: { city: { type: "string" }, date: { type: "string" } }, required: ["city"] },
    },
  },
];

async function completeStream(messages, tools) {
  const body = { model: MODEL, messages, tools, stream: true, temperature: 0.2 };
  const r = await fetch(URL, { method: "POST", headers: HEADERS, body: JSON.stringify(body) });
  if (!r.ok) return { error: r.status, body: (await r.text()).slice(0, 300) };
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const reasoning = [];
  const content = [];
  let finish = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      try {
        const ev = JSON.parse(data);
        const delta = ev.choices?.[0]?.delta ?? {};
        if (typeof delta.reasoning_content === "string" && delta.reasoning_content) reasoning.push(delta.reasoning_content);
        if (typeof delta.content === "string" && delta.content) content.push(delta.content);
        if (ev.choices?.[0]?.finish_reason) finish = ev.choices[0].finish_reason;
      } catch {}
    }
  }
  return { reasoning: reasoning.join(""), content: content.join(""), finish, reasoning_deltas: reasoning.length };
}

async function turn1() {
  const r = await fetch(URL, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "What's the weather like in London today?" }],
      tools: TOOLS,
      stream: false,
    }),
  });
  const json = await r.json();
  return json.choices[0].message;
}

// pi serializer simulation: native message -> wire message.
// With signature on the thinking block, pi replays the REAL text as
// reasoning_content (openai-completions convertMessages behavior).
function wireFromNative(native) {
  const messages = [];
  for (const m of native) {
    if (m.role === "user") {
      messages.push({ role: "user", content: m.content.map((b) => b.text).join("") });
    } else if (m.role === "assistant") {
      const asst = { role: "assistant", content: m.content.filter((b) => b.type === "text").map((b) => b.text).join("") };
      const tcs = m.content
        .filter((b) => b.type === "toolCall")
        .map((b) => ({ id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.arguments) } }));
      if (tcs.length) asst.tool_calls = tcs;
      const thinking = m.content.find((b) => b.type === "thinking");
      if (thinking && thinking.thinkingSignature === "reasoning_content" && thinking.thinking.trim()) {
        asst.reasoning_content = thinking.thinking; // pi convertMessages: assistantMsg[signature] = text
      }
      messages.push(asst);
    } else if (m.role === "toolResult") {
      messages.push({ role: "tool", tool_call_id: m.toolCallId, content: m.content[0].text });
    }
  }
  return { model: MODEL, messages, tools: TOOLS };
}

const t1 = await turn1();
const realReasoning = t1.reasoning_content ?? "";
console.log(`turn1: reasoning=${realReasoning.length} chars, calls=${(t1.tool_calls ?? []).map((t) => t.function.name).join(",")}`);
if (!t1.tool_calls?.length) throw new Error("no tool call in turn 1");

const toolMsgs = [];
for (const tc of t1.tool_calls) {
  const args = JSON.parse(tc.function.arguments ?? "{}");
  const result = tc.function.name === "get_date" ? { date: "2026-09-02" } : { temp: 28, sky: "sunny", city: args.city ?? "London" };
  toolMsgs.push({ role: "tool", tool_call_id: tc.id, content: JSON.stringify(result) });
}

// native stored messages: thinking block WITHOUT signature (simulates a
// session where the signature did not survive -> pi drops the reasoning)
const nativeBase = [
  { role: "user", content: [{ type: "text", text: "What's the weather like in London today?" }] },
  {
    role: "assistant",
    content: [
      { type: "thinking", thinking: realReasoning },
      { type: "text", text: t1.content ?? "" },
      ...t1.tool_calls.map((tc) => ({ type: "toolCall", id: tc.id, name: tc.function.name, arguments: JSON.parse(tc.function.arguments ?? "{}") })),
    ],
  },
  ...toolMsgs.map((m) => ({ role: "toolResult", toolCallId: m.tool_call_id, toolName: "x", content: [{ type: "text", text: m.content }], isError: false })),
];

const REP = 3;
const conditions = {
  "A) no extension (no reasoning_content)": null,
  "B) wire fix only (placeholder ' ')": "wire",
  "C) native+wire fix (real text replay)": "both",
};

const results = {};
for (const [name, mode] of Object.entries(conditions)) {
  const samples = [];
  for (let i = 0; i < REP; i++) {
    let native = JSON.parse(JSON.stringify(nativeBase));
    if (mode === "both") {
      const r = fixNativeMessagesForDeepSeek(native);
      native = r.messages;
    }
    let wire = wireFromNative(native);
    if (mode === "wire" || mode === "both") {
      const fixed = fixWirePayloadForDeepSeek(wire);
      if (mode === "wire") {
        if (!fixed) throw new Error(`expected wire fix to touch payload (${name})`);
      }
      if (fixed) wire = fixed; // C: no-op when real text already replayed
    }
    const res = await completeStream(wire.messages, wire.tools);
    if (res.error) {
      samples.push({ error: `${res.error}: ${res.body}` });
      break;
    }
    samples.push({ chars: res.reasoning.length, deltas: res.reasoning_deltas, finish: res.finish });
  }
  results[name] = samples;
}

console.log("\n=== results (median of " + REP + " reps) ===");
for (const [name, samples] of Object.entries(results)) {
  if (samples.some((s) => s.error)) {
    console.log(`${name}: ERROR ${samples.find((s) => s.error).error}`);
    continue;
  }
  const chars = samples.map((s) => s.chars).sort((a, b) => a - b);
  const deltas = samples.map((s) => s.deltas).sort((a, b) => a - b);
  const median = (arr) => arr[Math.floor(arr.length / 2)];
  console.log(`${name}: reasoning median=${median(chars)} chars / ${median(deltas)} deltas | finishes=${samples.map((s) => s.finish).join(",")}`);
}
