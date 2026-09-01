/**
 * Replays the EXACT wire payload pi sent at 22:09:53 (the failing request)
 * against litellm.private, N times, to see if the payload shape causes
 * "Connection closed" or if it is transient upstream flakiness.
 *
 * Reconstructs pi's openai-completions serialization:
 *  - thinking block with signature "reasoning_content" -> reasoning_content
 *  - text blocks -> content
 *  - toolCall blocks -> tool_calls
 *  - toolResult -> role tool
 *  - wire fix: force " " on assistants missing reasoning_content
 *  - thinking: {type:"enabled"} + reasoning_effort high (PI_REASONING_LEVEL)
 */
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// dist lives at the repo root: <repo>/tools/diagnostics -> <repo>/dist
const DIST_INDEX = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "dist", "index.js");
const { fixWirePayloadForDeepSeek } = await import(DIST_INDEX);

const KEY = process.env.GATEWAY_API_KEY;
const URL = "http://litellm.private/v1/chat/completions";
const MODEL = "deepseek/deepseek-v4-flash";
const HEADERS = {
  Authorization: `Bearer ${KEY}`,
  "Content-Type": "application/json",
  "X-OpenWebUI-User-Name": "pi",
};

// ---- load native messages from the failing session ----
const sessionPath = process.env.HOME + "/.pi/agent/sessions/--work--/2026-09-01T22-07-23-438Z_01a05f03-3f2e-7668-8edb-80f215c42787.jsonl";
const native = [];
for (const line of readFileSync(sessionPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  let ev;
  try { ev = JSON.parse(line); } catch { continue; }
  if (ev.type !== "message") continue;
  const m = ev.message;
  if (m.role === "user" || m.role === "assistant" || m.role === "toolResult") native.push(m);
}

// history up to the failing request = messages[0..15] + new user msg (20)
const history = native.slice(0, 16);
const newUser = native[20];

function toWire(m) {
  const c = Array.isArray(m.content) ? m.content : [];
  if (m.role === "user") {
    return { role: "user", content: c.map((b) => b.text).join("") };
  }
  if (m.role === "assistant") {
    const asst = { role: "assistant", content: c.filter((b) => b.type === "text").map((b) => b.text).join("") };
    const tcs = c
      .filter((b) => b.type === "toolCall")
      .map((b) => ({ id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.arguments ?? {}) } }));
    if (tcs.length) asst.tool_calls = tcs;
    const thinking = c.find((b) => b.type === "thinking");
    if (thinking && thinking.thinkingSignature === "reasoning_content" && typeof thinking.thinking === "string" && thinking.thinking.trim()) {
      asst.reasoning_content = thinking.thinking; // pi serializer replay
    }
    return asst;
  }
  if (m.role === "toolResult") {
    return { role: "tool", tool_call_id: m.toolCallId, content: m.content[0]?.text ?? "" };
  }
  return null;
}

const wireMessages = [...history.map(toWire), toWire(newUser)].filter(Boolean);
const wire = {
  model: MODEL,
  messages: wireMessages,
  tools: [
    { type: "function", function: { name: "bash", description: "Run a command", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
    { type: "function", function: { name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  ],
  stream: true,
  temperature: 0.2,
  thinking: { type: "enabled" },
  reasoning_effort: "high",
};

const fixed = fixWirePayloadForDeepSeek(JSON.parse(JSON.stringify(wire)));
console.log("wire fix applied:", !!fixed);
const payload = fixed ?? wire;

// sanity: what reasoning_content does each assistant carry?
payload.messages.forEach((m, i) => {
  if (m.role === "assistant") {
    console.log(`asst[${i}]: reasoning=${(m.reasoning_content ?? "").length} chars, tool_calls=${(m.tool_calls ?? []).length}`);
  }
});
console.log("payload size:", JSON.stringify(payload).length, "bytes,", payload.messages.length, "messages");

async function streamOnce(n) {
  const started = Date.now();
  let r;
  try {
    r = await fetch(URL, { method: "POST", headers: HEADERS, body: JSON.stringify(payload) });
  } catch (e) {
    return { kind: "fetch-error", err: String(e), ms: Date.now() - started };
  }
  if (!r.ok) {
    const body = (await r.text()).slice(0, 300);
    return { kind: "http", status: r.status, body, ms: Date.now() - started };
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", reasoning = 0, content = 0, events = 0, finish = null, firstChunkMs = null;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (firstChunkMs === null) firstChunkMs = Date.now() - started;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        events++;
        try {
          const ev = JSON.parse(data);
          const delta = ev.choices?.[0]?.delta ?? {};
          if (typeof delta.reasoning_content === "string" && delta.reasoning_content) reasoning += delta.reasoning_content.length;
          if (typeof delta.content === "string" && delta.content) content += delta.content.length;
          if (ev.choices?.[0]?.finish_reason) finish = ev.choices[0].finish_reason;
        } catch {}
      }
    }
  } catch (e) {
    return { kind: "midstream-error", err: String(e), reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs };
  }
  return { kind: "ok", reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs };
}

const REPS = Number(process.env.REPS ?? 8);
let failures = 0;
for (let i = 1; i <= REPS; i++) {
  const res = await streamOnce(i);
  if (res.kind === "ok") {
    console.log(`ok   #${i}: reasoning=${res.reasoning} chars, content=${res.content}, finish=${res.finish}, ttfb=${res.firstChunkMs}ms, total=${res.ms}ms`);
  } else {
    failures++;
    console.log(`FAIL #${i}: ${res.kind} ${res.err ?? res.status + " " + (res.body ?? "")} ttfb=${res.firstChunkMs ?? "-"}ms total=${res.ms}ms`);
  }
}
console.log(`\n${REPS - failures}/${REPS} ok, ${failures} failures`);
