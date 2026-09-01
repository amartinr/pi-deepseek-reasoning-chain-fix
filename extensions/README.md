# deepseek-reasoning-chain

Pi Coding Agent extension that fixes the **DeepSeek reasoning-chaining
contract** on tool-calling turns.

## The problem

DeepSeek's thinking mode returns chain-of-thought in `reasoning_content`.
The [official docs](https://api-docs.deepseek.com/guides/thinking_mode#tool-calls)
are explicit about the contract:

> For requests carrying the `tools` parameter, the `reasoning_content` must be
> fully passed back to the API in all subsequent requests — even for turns
> where the model did not perform a tool call. If your code does not correctly
> pass back `reasoning_content`, the API will return a 400 error.

With `tools` in scope the CoT **is** concatenated into the context, so each
continuation can continue the previous chain. Consequences of breaking it:

- **Missing field** → 400 against the raw API; through gateways (LiteLLM)
  a blank placeholder is injected with a warning — the model gets an empty
  chain.
- **Empty value `""`** → treated as absent by LiteLLM/DeepSeek; same blank
  chain.
- **`thinking: {"type": "disabled"}` on a continuation** → hard kill-switch:
  the model produces **0 reasoning deltas** (verified live).

## What pi already does (and the gaps)

The openai-completions serializer:

1. Replays the **real** reasoning text as `reasoning_content` — but only when
   the stored `thinking` content block carries a recognized
   `thinkingSignature` (set at stream time). A block without a signature falls
   back to nothing (e.g. sessions resumed from disk where the extra field did
   not survive).
2. Forces `reasoning_content = ""` on assistant messages — but only when its
   own DeepSeek detection fires (`provider === "deepseek"` or a baseUrl
   containing `deepseek.com`). Behind a gateway such as LiteLLM at
   `litellm.private` (model `deepseek/deepseek-v4-flash`) that detection
   **never fires**.
3. Sends `thinking: {"type": "disabled"}` when reasoning is off, which kills
   reasoning on tool-call continuations.

## The fix (3 hooks, same layers as the validated agent_loop_guard pipe)

| Hook | Layer | Fix |
|------|-------|-----|
| `context` | native messages, before serialization | stamp `thinkingSignature = "reasoning_content"` on non-empty thinking blocks → pi's serializer replays the **real text** on continuations (idempotent, covers lost signatures) |
| `before_provider_request` | wire payload, last hop before the endpoint | in tool scope, force **non-empty** `reasoning_content` (`" "`) on every assistant message; strip `thinking: disabled` when the conversation has been reasoning |
| `message_end` | return path | normalize the finalized assistant message so the **stored** reasoning stays replayable by the next continuation |

Scope detection covers direct DeepSeek, DeepSeek behind any gateway that keeps
the `deepseek/` model prefix (LiteLLM), and baseUrls containing
`deepseek.com`. Everything is deterministic and fails open.

## Validation

- `verify-extension.mjs` — loads the extension with pi's own loader (jiti)
  and unit-checks all fix functions (idempotence, scope, no-touch cases).
- `verify-live.mjs` — live A/B against a real LiteLLM gateway
  (`deepseek/deepseek-v4-flash`): no fix vs wire-fix-only vs native+wire.
  Consistent finding across runs: placeholder-only can drop continuation
  reasoning to **0 deltas**; real-text replay keeps the chain alive.

## Install

As a pi package (built artifact, `dist/` only):

```bash
pi install pi-deepseek-reasoning-chain@0.1.0   # once published
```

Or install from this directory:

```bash
pi install /work/extensions
```

Manual copy (single-file source, for development):

```bash
cp src/index.ts ~/.pi/agent/extensions/deepseek-reasoning-chain.ts
```

Then `/reload` in pi (or restart). Optional:
`PI_DEEPSEEK_REASONING_EXTRA` env var adds comma-separated model prefixes to
treat as DeepSeek.

## Development

```bash
npm install        # dev deps: typescript, pi types
npm run build      # tsc: src/ -> dist/
npm run verify     # unit checks against the built artifact
node verify-live.mjs   # live A/B against a LiteLLM gateway
npm pack --dry-run # inspect the published tarball (dist only, no src)
```
