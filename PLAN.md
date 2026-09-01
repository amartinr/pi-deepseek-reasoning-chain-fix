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

## P0 — ~~thinking:disabled strip~~ — REVERTED (false premise)

**Original claim:** strip `thinking: {type:"disabled"}` on tool-call
continuations, matching the agent_loop_guard pipe (Open WebUI injects the
marker on continuations).

**Reversal (v0.2.1):** the premise does not hold for pi. pi sends
`thinking: disabled` only when the **user** chose thinking off; there is no
Open WebUI-style auto-injection to counteract. Stripping the marker would
override explicit user intent — the user must be able to choose a
non-reasoning model. The strip was removed entirely; the wire layer keeps
only the `reasoning_content` forcing (the contract, not a choice).

- [x] Strip removed from `fixWirePayloadForDeepSeek` (thinking preserved).
- [x] Tests updated: `thinking` preserved in both modes; live check:
      user-disabled thinking + `" "` forcing satisfies the contract (no
      400) without reasoning.

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

- [x] Scope is config-driven: no heuristic matcher remains; `modelsMatch()`
      covers exact/prefix/case-insensitive and the empty-list inert case.
- [x] Unit tests: exact, prefix, case-insensitive, bare id not configured,
      empty list never matches, non-deepseek excluded; `loadConfig` covers
      valid, malformed and missing files (all fail-open).

## P3 — test coverage for the hardened behavior

`verify-extension.mjs` covers the happy paths; add:

- [x] `thinking: disabled` is PRESERVED in every case — real reasoning
      (no-op), placeholder forcing (contract met, marker intact), plain
      non-tool chat (untouched), and compliance mode (wire active,
      thinking kept).
- [x] Single-pass equivalence: output identical to the old 3-pass logic on
      the existing fixtures (regression guard).
- [x] String `content` robustness (P1).
- [x] Prefix-cache stability: running the fix twice on the same input
      produces a byte-identical payload (determinism guard).

## P3 — observability

- [x] Resolved with the config-driven scope: the load log always reports the
      active model ids or the inert state, so a silent miss is impossible
      (the original proposal — an env-gated startup log — was superseded;
      gating would hide the activation signal).

## Out of scope (deliberate)

- No network I/O, retries, or caches — the extension transforms in-memory
  payloads only; nothing to add there.
- No `before_provider_headers` involvement.

## Definition of done

- [x] All P0/P1 items implemented with unit tests in `verify-extension.mjs`.
- [x] `npm run build` + `npm run verify` pass; live A/B (`verify-live.mjs`)
      re-run against the gateway for the strip semantics — the P0 strip
      turns a hard HTTP 400 (`reasoning_content must be passed back`) into
      working reasoning (32/36 deltas).
- [x] `npm version patch` → v0.1.2, published, changelog via commit history.
- [x] Branch merged to `master` only after the live check passes.

## Post-plan (shipped after the merge)

- [x] **v0.2.0 — `replayReasoning` knob.** Config option to switch between
      full chaining (real text replay) and compliance-only (`" "`
      placeholder). Native layer on/off; wire layer always active.
- [x] **v0.2.1 — strip reverted (false premise).** The `thinking:disabled`
      strip was removed: pi sends the marker only when the user chose
      thinking off (no Open WebUI-style injection), so stripping would
      override user intent. `thinking` is now preserved in every path;
      only the `reasoning_content` forcing remains. Live-verified:
      user-disabled thinking + `" "` forcing satisfies the contract (no
      400).
