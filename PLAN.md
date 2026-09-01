# PLAN — deepseek-reasoning-chain-fix improvements

Branch: `improvements` (based on `master` @ `fa26703`, v0.1.1).

All items are derived from a full review of `src/index.ts` (correctness,
efficiency, choke points, timeouts, resource consumption).

## Baseline review verdict

- No I/O, timers, or state in any hook → **no timeouts to manage, no
  concurrency choke points, no resource accumulation**. All three hooks are
  pure, synchronous, deterministic, fail-open.
- Performance is O(n) per request over the messages array, with small
  constants. The only meaningful cost is the 3-pass scan on very long
  histories (contextWindow up to 1M tokens).
- The changes below are hardening + one semantic fix, not a rewrite.

## P0 — fix the `thinking:disabled` strip semantics

**Current behavior (subtle interaction):** `fixWirePayloadForDeepSeek` forces
`reasoning_content` on all assistants *before* evaluating
`historyHasReasoning`. When the real-text replay failed and everything was
placeholder-forced (`" "`), the history no longer contains detectable
reasoning → the strip never fires → the kill-switch stays on exactly in the
degraded scenario the extension exists to prevent.

**Proposal:** strip `thinking: {type:"disabled"}` whenever the request is a
tool-call continuation (assistant with `tool_calls` in history, i.e. the
DeepSeek contract applies), matching the validated agent_loop_guard pipe.
A continuation with thinking disabled is broken by definition for DeepSeek
(0 reasoning deltas, verified live) — the user's "disabled from the start"
intent is only meaningful for non-tool turns.

- [x] Recompute the continuation flag from the history *before* any forcing
      (single pass, see P1).
- [x] Update the function docstring to state the semantics explicitly.

## P1 — single-pass wire normalization

`before_provider_request` currently runs three linear scans over `messages`:

1. `isToolScope(payload)` — `.some(...)`
2. forcing loop — every assistant
3. `historyHasReasoning(messages)` — `.some(...)`

**Proposal:** one pass computing: `inToolScope`, `hasAssistantToolCalls`
(continuation), applying the forcing in the same iteration. This halves the
constant factor on 1M-token histories.

- [x] Merge into a single loop; keep the pure-function signature
      (`fixWirePayloadForDeepSeek(payload) -> payload | undefined`).
- [x] Preserve determinism (stable serialization, prefix cache untouched).
      Mutations are applied only after the scope decision, so out-of-scope
      payloads are never touched (regression-tested).

## P1 — defensive content guards

`fixNativeMessagesForDeepSeek` and `fixFinalizedMessageForDeepSeek` iterate
`msg.content ?? []` / `message.content ?? []`. If `content` were ever a
string (provider/format edge), the loop iterates characters silently.

- [x] Guard both with `Array.isArray(...)` checks; non-array content → skip.
- [x] Same guard in the new single-pass wire loop for `tool_calls`.
      Regression-tested: string content never crashes nor mutates.

## P2 — config-driven scope (replaces heuristic detection)

**Resolution:** the heuristics (provider/baseUrl/id-prefix sniffing) were
removed entirely. The scope is now explicit configuration:

- `~/.pi/agent/extensions/pi-deepseek-reasoning-chain-fix/config.json` with
  `{ "models": [...] }` — the model ids (exact or prefix match,
  case-insensitive).
- **Empty/missing list → extension inert** (safe default: forcing
  DeepSeek-specific fields on arbitrary models is dangerous).
- `PI_DEEPSEEK_REASONING_CONFIG` overrides the path (tests, custom setups).
- Fail-open on malformed files; the load log reports the active ids or the
  inert state (no more silent misses).
- `isDeepSeekModel`/`PI_DEEPSEEK_REASONING_EXTRA` removed; the matcher is
  `modelsMatch()` and the loader `loadConfig()` (both exported and tested).

- [x] Widen the matcher without false positives on non-DeepSeek models.
- [x] Unit tests for: bare `deepseek-v4-flash` id, custom provider,
      false-positive guard (e.g. `deepseek-models-test` on a random provider
      stays excluded unless matched by the extra env list).

## P3 — test coverage for the hardened behavior

`verify-extension.mjs` covers the happy paths; add:

- [x] Strip fires when history has real reasoning AND on any continuation
      (P0 semantics).
- [x] Strip does NOT fire on a plain non-tool chat (no tool scope).
- [x] Single-pass equivalence: output identical to the old 3-pass logic on
      the existing fixtures (regression guard).
- [x] String `content` robustness (P1).
- [x] Prefix-cache stability: running the fix twice on the same input
      produces a byte-identical payload (determinism guard).

## P3 — observability

- [ ] Gate the startup `console.log` behind
      `PI_DEEPSEEK_REASONING_LOG=1` (quiet by default); keep a one-time
      log on the first request where scope detection matches, so a silent
      miss (P2) is visible in the logs.

## Out of scope (deliberate)

- No network I/O, retries, or caches — the extension transforms in-memory
  payloads only; nothing to add there.
- No `before_provider_headers` involvement.

## Definition of done

- All P0/P1 items implemented with unit tests in `verify-extension.mjs`.
- `npm run build` + `npm run verify` pass; live A/B (`verify-live.mjs`)
  re-run against the gateway for the strip semantics.
- `npm version patch` → v0.1.2, published, changelog via commit history.
- Branch merged to `master` only after the live check passes.
