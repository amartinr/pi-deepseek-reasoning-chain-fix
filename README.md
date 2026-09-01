# DeepSeek Reasoning Chain

Fixes the **DeepSeek reasoning-chaining contract** on tool-calling turns for
the Pi Coding Agent, backed by a validation harness and live probes against a
real LiteLLM gateway.

## The problem

DeepSeek's thinking mode returns chain-of-thought in `reasoning_content`. The
[official API docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)
define the contract:

> For requests carrying the `tools` parameter, the `reasoning_content` must be
> fully passed back to the API in all subsequent requests — even for turns
> where the model did not perform a tool call. If your code does not correctly
> pass back `reasoning_content`, the API will return a 400 error.

When `tools` is in scope, the chain-of-thought **is** concatenated into the
context, so each tool-call continuation can continue the previous reasoning.
Breaking the contract has three failure modes, all verified live:

| Failure | Symptom |
|---------|---------|
| `reasoning_content` missing | 400 on the raw API; via LiteLLM a blank placeholder is injected (with a server warning) |
| `reasoning_content` empty (`""`) | treated as absent — blank chain, silently degraded multi-turn reasoning |
| `thinking: {"type":"disabled"}` on a continuation | hard kill-switch — **0 reasoning deltas** |

## Why an extension is needed (what pi already does)

The openai-completions serializer in pi:

1. Replays the **real** reasoning text as `reasoning_content` — but only when
   the stored `thinking` content block carries a recognized
   `thinkingSignature`. Blocks without one fall back to nothing.
2. Forces `reasoning_content = ""` on assistant messages — but only when its
   own DeepSeek detection fires (`provider === "deepseek"` or a baseUrl
   containing `deepseek.com`). Behind a gateway like LiteLLM at
   `litellm.private` (`api: openai-completions`, model
   `deepseek/deepseek-v4-flash`) that detection **never fires**.
3. Sends `thinking: {"type": "disabled"}` when reasoning is off, which kills
   reasoning on tool-call continuations.

This extension closes all three gaps. Its design mirrors the validated
`agent_loop_guard` Open WebUI pipe (same three layers, translated to pi
hooks).

## Repository layout

```
.
├── extensions/
│   ├── deepseek-reasoning-chain.ts   # the pi extension (single file)
│   ├── verify-extension.mjs          # unit checks (jiti load + pure functions)
│   ├── verify-live.mjs               # live A/B against a LiteLLM gateway
│   └── README.md                     # extension-specific docs
├── tests/
│   ├── live_reasoning_probe.py       # live probe (turn 1 + 3 continuation conditions)
│   └── live_ab_reasoning.py          # repeatable A/B + thinking:disabled test
├── probes/litellm/
│   └── owui_misc_stub.py             # real convert_output_to_messages() from
│                                     # open-webui v0.11.1 (offline replay tests)
└── alg-sim/                          # self-contained unit-test harness
    ├── agent_loop_guard/             # pipe copy + its 59 unit tests
    │   ├── conftest.py               # fake open_webui package for patch tests
    │   └── tests/
    └── fake_owui/                    # minimal open_webui import stub
```

## The extension — how it works

The extension is an npm package (`extensions/`). Packaging convention:

- **GitHub**: the repo holds `extensions/src/index.ts` (TypeScript source).
  `dist/` is gitignored and never pushed.
- **npm**: the tarball ships only the built `dist/` (`files: ["dist"]` plus
  `.npmignore` as an explicit backstop) — the `src/` content is never
  included.
- Build: `npm run build` (tsc) compiles `src/` → `dist/`; `prepublishOnly`
  rebuilds before publish; `pi.extensions: ["./dist/index.js"]` declares the
  pi extension entry point.

Three hooks, one per layer of the fix:

| Hook | Layer | Fix |
|------|-------|-----|
| `context` | native messages, before serialization | stamp `thinkingSignature = "reasoning_content"` on non-empty thinking blocks → pi's serializer replays the **real text** on continuations (idempotent; covers signatures lost on session resume) |
| `before_provider_request` | wire payload, last hop before the endpoint | in tool scope, force **non-empty** `reasoning_content` (`" "`) on every assistant message; strip `thinking: disabled` when the conversation has been reasoning |
| `message_end` | return path | normalize the finalized assistant message so the **stored** reasoning stays replayable by the next continuation |

Scope detection matches `provider === "deepseek"`, baseUrls containing
`deepseek.com`, and model ids with the `deepseek/` or `deepseek-` prefix
(covers gateways such as LiteLLM that keep the prefix). All transformations
are deterministic, idempotent, and fail open.

## Validation

### Unit tests (offline)

```
python3 -m venv /tmp/alg-venv
/tmp/alg-venv/bin/pip install pytest httpx pydantic
cd alg-sim/agent_loop_guard && /tmp/alg-venv/bin/python -m pytest tests/ -q
# 59 passed (forcing, replay-vs-no-replay, attached-files cleanup)
```

The reasoning-replay tests use the **real** `convert_output_to_messages()`
from open-webui v0.11.1 (vendored in `probes/litellm/owui_misc_stub.py`), so
they prove the patch changes Open WebUI's actual history reconstruction, not
a mock.

### Extension unit checks

```
cd extensions && node verify-extension.mjs
# loads via jiti (pi's loader), scope detection, idempotence, no-touch cases
```

### Live A/B (against a real gateway)

```
cd extensions && node verify-live.mjs
# 3 conditions x 3 reps: no fix / wire-only / native+wire
```

Consistent findings across runs on `deepseek/deepseek-v4-flash`:

- placeholder-only (`" "`) can drop continuation reasoning to **0 deltas**
  (the silent degradation);
- real-text replay (native fix) keeps the chain alive — reasoning never 0,
  and at its best 4× richer with visible chain continuity;
- `thinking: disabled` on a continuation always yields **0 reasoning deltas**
  vs 17–38 without — the strip restores reasoning every time.

## Install

```bash
cp extensions/deepseek-reasoning-chain.ts ~/.pi/agent/extensions/
```

Then `/reload` in pi (or restart). Optional:
`PI_DEEPSEEK_REASONING_EXTRA` env var adds comma-separated model prefixes to
treat as DeepSeek.

## Security

The API key is never stored in the repo. Live probes read it from
`$LITELLM_API_KEY` or a `0600` file (`.gitignore` protects `*.key`, `.env`,
and all probe/harness directories from accidental commits).

## License

MIT
