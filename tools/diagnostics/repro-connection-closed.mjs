/**
 * Reproduce "DeepseekException - Connection closed" against litellm.private.
 * Simulates the exact pi flow: turn1 (tools, non-stream) -> continuation
 * with stream:true, assistant carrying reasoning_content (the contract).
 * Tries N reps to catch the intermittent mid-stream disconnect.
 */
const KEY = process.env.GATEWAY_API_KEY;
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

async function nonStream(messages, tools, extra = {}) {
  const r = await fetch(URL, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ model: MODEL, messages, tools, stream: false, ...extra }),
  });
  const text = await r.text();
  try {
    return { status: r.status, json: JSON.parse(text) };
  } catch {
    return { status: r.status, text: text.slice(0, 500) };
  }
}

async function stream(messages, tools, extra = {}, label = "") {
  const started = Date.now();
  let r;
  try {
    r = await fetch(URL, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: MODEL, messages, tools, stream: true, ...extra }),
      // no timeout: let the server hang if it wants
    });
  } catch (e) {
    return { kind: "fetch-error", err: String(e), ms: Date.now() - started, label };
  }
  if (!r.ok) {
    const body = (await r.text()).slice(0, 400);
    return { kind: "http", status: r.status, body, ms: Date.now() - started, label };
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let reasoning = 0, content = 0, events = 0, finish = null;
  let firstChunkMs = null;
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
    return { kind: "midstream-error", err: String(e), reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs, label };
  }
  return { kind: "ok", reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs, label };
}

// ---- turn 1: get an authentic tool-call assistant with reasoning_content ----
const t1 = await nonStream([{ role: "user", content: "What's the weather in London today? Use the tools." }], TOOLS);
const msg = t1.json?.choices?.[0]?.message;
if (!msg?.tool_calls?.length) {
  console.log("turn1 failed or no tool call:", JSON.stringify(t1).slice(0, 500));
  process.exit(1);
}
const realReasoning = msg.reasoning_content ?? "";
console.log(`turn1 ok: reasoning=${realReasoning.length} chars, calls=${msg.tool_calls.map((t) => t.function.name).join(",")}`);

const toolMsgs = msg.tool_calls.map((tc, i) => ({
  role: "tool",
  tool_call_id: tc.id,
  content: i === 0 ? JSON.stringify({ date: "2026-09-02" }) : JSON.stringify({ temp: 28, sky: "sunny", city: "London" }),
}));

function continuation(includeReasoning, placeholderOnly = false) {
  const asst = { role: "assistant", content: msg.content ?? "", tool_calls: msg.tool_calls };
  if (includeReasoning) {
    asst.reasoning_content = placeholderOnly ? " " : realReasoning;
  }
  return [
    { role: "user", content: "What's the weather in London today? Use the tools." },
    asst,
    ...toolMsgs,
    { role: "user", content: "Now summarize the weather for me, please." },
  ];
}

const REPS = 4;
console.log(`\n=== streaming continuation, ${REPS} reps per condition ===`);

for (const [label, messages] of [
  ["A) contract broken (no reasoning_content)", continuation(false)],
  ["B) placeholder ' ' only", continuation(true, true)],
  ["C) real reasoning replay", continuation(true, false)],
]) {
  for (let i = 0; i < REPS; i++) {
    const res = await stream(messages, TOOLS, {}, `${label} #${i + 1}`);
    if (res.kind === "ok") {
      console.log(`ok    | ${label} #${i + 1}: reasoning=${res.reasoning} chars, content=${res.content}, events=${res.events}, finish=${res.finish}, ttfb=${res.firstChunkMs}ms, total=${res.ms}ms`);
    } else {
      console.log(`FAIL  | ${label} #${i + 1}: ${res.kind} ${res.err ?? res.status + " " + (res.body ?? "")} (ttfb=${res.firstChunkMs ?? "-"}ms, total=${res.ms}ms)`);
    }
  }
}
