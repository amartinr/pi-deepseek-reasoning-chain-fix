/**
 * Stress: long streams + concurrent streams to trigger the intermittent
 * "Connection closed". Also probes the gateway version/headers.
 */
const KEY = process.env.GATEWAY_API_KEY;
const URL = "http://litellm.private/v1/chat/completions";
const MODEL = "deepseek/deepseek-v4-flash";
const HEADERS = { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" };

// gateway version probe
const h = await fetch("http://litellm.private/", { headers: { Authorization: `Bearer ${KEY}` } });
console.log("gateway headers:", h.headers.get("server"), "| version:", (await h.text()).slice(0, 120));

async function streamOne(prompt, extra = {}, timeoutMs = 180_000) {
  const started = Date.now();
  const ac = new AbortController();
  const to = setTimeout(() => ac.abort(), timeoutMs);
  let r;
  try {
    r = await fetch(URL, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: prompt }], stream: true, ...extra }),
      signal: ac.signal,
    });
  } catch (e) {
    clearTimeout(to);
    return { kind: "fetch-error", err: String(e), ms: Date.now() - started };
  }
  if (!r.ok) {
    clearTimeout(to);
    return { kind: "http", status: r.status, body: (await r.text()).slice(0, 200), ms: Date.now() - started };
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", reasoning = 0, content = 0, events = 0, finish = null, firstChunkMs = null, lastGap = 0, lastChunkAt = started;
  try {
    while (true) {
      const { done, value } = await reader.read();
      const now = Date.now();
      if (done) break;
      if (firstChunkMs === null) firstChunkMs = now - started;
      const gap = now - lastChunkAt;
      if (gap > lastGap) lastGap = gap;
      lastChunkAt = now;
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
    clearTimeout(to);
    return { kind: "midstream-error", err: String(e), reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs, lastGap };
  }
  clearTimeout(to);
  return { kind: "ok", reasoning, content, events, finish, ms: Date.now() - started, firstChunkMs, lastGap };
}

// ---- long-generation test: keep the stream open for a while ----
console.log("\n=== long stream (max effort, long output) ===");
const long = await streamOne(
  "Write a very long, detailed essay (about 3000 words) explaining the history of computing, " +
  "from mechanical calculators to modern AI. Go into depth on every era. Do not stop early.",
  { thinking: { type: "enabled" }, reasoning_effort: "max", max_tokens: 8000 }
);
console.log("long:", long.kind, long.err ?? `reasoning=${long.reasoning} content=${long.content} finish=${long.finish} ttfb=${long.firstChunkMs}ms total=${long.ms}ms lastGap=${long.lastGap}ms`);

// ---- concurrency test ----
console.log("\n=== 6 concurrent short streams ===");
const prompts = [
  "What is 2+2? Answer briefly.",
  "What is the capital of France?",
  "Name 3 colors.",
  "What is pi? One sentence.",
  "Count from 1 to 10.",
  "What day comes after Sunday?",
];
const results = await Promise.all(prompts.map((p) => streamOne(p, {}, 60_000)));
results.forEach((r, i) => {
  console.log(`#${i}: ${r.kind} ${r.err ?? `reasoning=${r.reasoning} content=${r.content} finish=${r.finish} total=${r.ms}ms`}`);
});
