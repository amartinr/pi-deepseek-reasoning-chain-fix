# Design Document — pi-deepseek-reasoning-chain-fix

Version: 0.1.1 · Branch: `improvements` · Companion docs: `README.md` (user),
`PLAN.md` (hardening backlog).

## 1. Purpose

A Pi Coding Agent extension that keeps DeepSeek reasoning working across
tool-calling turns. It enforces the DeepSeek `reasoning_content` contract on
every outbound request and preserves the real reasoning text between
continuations, so the model can continue its previous chain of thought
instead of re-deriving it (or losing it entirely).

## 2. Background: the DeepSeek contract

DeepSeek's thinking mode returns chain-of-thought in `reasoning_content`,
alongside `content`. The official API documentation is explicit about
tool-calling histories:

> For requests carrying the `tools` parameter, the `reasoning_content` must
> be fully passed back to the API in all subsequent requests — even for turns
> where the model did not perform a tool call. If your code does not
> correctly pass back `reasoning_content`, the API will return a 400 error.

Consequences (each verified live against a real gateway):

| Failure mode | Symptom |
|--------------|---------|
| Field missing on an assistant message | 400 on the raw API; through LiteLLM a single-space placeholder is injected with a server warning (`transformation.py`) |
| Field present but empty (`""`) | Treated as absent — blank chain, silently degraded multi-turn reasoning |
| `thinking: {"type":"disabled"}` on a continuation | Hard kill-switch: **0 reasoning deltas** on that turn |

Two more contract facts drive the design:

- With `tools` in scope, the chain-of-thought **is concatenated into the
  context** — this is what makes chaining possible at all.
- Thinking mode ignores `temperature`, `top_p`, and the other sampling
  parameters (silently).

## 3. Analysis: what pi already does, and the gaps

Pi's `openai-completions` serializer (`convertMessages`) was inspected in the
installed bundle. Three relevant behaviors:

1. **Real-text replay is signature-gated.** A stored `thinking` content block
   is replayed as `reasoning_content` only when it carries a recognized
   `thinkingSignature` (`reasoning_content` / `reasoning` / `reasoning_text`),
   set at stream time. A block without a signature replays nothing.
2. **Empty-string forcing is detection-gated.** Pi forces
   `reasoning_content = ""` on assistant messages, but only when its own
   DeepSeek detection fires (`provider === "deepseek"` or a baseUrl
   containing `deepseek.com`). Behind a gateway such as LiteLLM at
   `litellm.private` (provider `litellm`, `api: openai-completions`, model
   `deepseek/deepseek-v4-flash`) that detection **never fires** — nothing is
   forced, the field is missing, and the gateway injects a blank placeholder.
3. **`thinking: disabled` is sent when reasoning is off** and kills
   reasoning on tool-call continuations.

Gaps this extension closes:

- **G1** — Real reasoning text lost on continuations when the signature is
  missing (resumed sessions, migrated histories).
- **G2** — No `reasoning_content` at all behind gateways that pi does not
  recognize as DeepSeek (LiteLLM-style routing).
- **G3** — `thinking: disabled` kill-switch on tool-call continuations.

## 4. Design goals and non-goals

Goals:

- Enforce the contract at the **last hop before the endpoint** (wire layer).
- Preserve the **real** reasoning text whenever available; use a non-empty
  placeholder only as a fallback.
- Deterministic, idempotent, fail-open, stateless.
- Zero I/O, zero timers, zero persistent state in the hooks.

Non-goals:

- No model-side prompting or content rewriting.
- No retry/caching logic — the extension transforms in-memory payloads only.
- No changes to pi's display or session format.

## 5. Architecture: three hooks, three layers

```
pi session (deepseek model in tool scope)
  │
  ├─ context (native messages, deep copy)
  │    └─ stamp thinkingSignature="reasoning_content" on thinking blocks
  │       → pi's serializer replays the REAL text (G1)
  │
  ├─ before_provider_headers
  │
  ├─ before_provider_request (wire payload, last hop before the endpoint)
  │    └─ in tool scope: force non-empty reasoning_content (" ") on every
  │       assistant; strip thinking:disabled on continuations (G2, G3)
  │
  ├─ endpoint (DeepSeek direct / LiteLLM gateway)
  │
  └─ message_end (finalized assistant message)
       └─ stamp thinkingSignature on the STORED message so the next
          continuation can replay it (persistence safety net, G1)
```

### 5.1 `context` — native layer (G1)

`fixNativeMessagesForDeepSeek(messages)` sets
`thinkingSignature = "reasoning_content"` on every non-empty thinking block
of every assistant message. This is the pi-native equivalent of the
`agent_loop_guard` pipe's `get_reasoning_format` monkey-patch: it makes the
built-in serializer replay the real chain-of-thought text instead of dropping
it. Idempotent: re-running over an already-stamped history changes nothing.

### 5.2 `before_provider_request` — wire layer (G2, G3)

`fixWirePayloadForDeepSeek(payload)`:

1. Skips unless the request is in tool scope — `tools` present (even `[]`)
   **or** an assistant with `tool_calls` in the history (the contract is
   driven by history content, not by this request's `tools`).
2. Forces `reasoning_content = " "` (single space — the exact placeholder
   the gateway would inject anyway, made explicit and non-empty) on every
   assistant message that lacks it or carries it empty. Real text, when the
   serializer already replayed it, is never touched.
3. Strips `thinking: {"type": "disabled"}` on tool-call continuations (see
   §6, decision D4).

Returns the payload only when something changed (`undefined` otherwise), per
the hook contract: "returning `undefined` keeps the payload unchanged".

### 5.3 `message_end` — return path (G1 persistence)

`fixFinalizedMessageForDeepSeek(message)` stamps the signature on the
finalized assistant message that pi is about to store. This guarantees the
stored reasoning stays replayable by the next continuation regardless of what
the session persistence does with extra block fields.

## 6. Key design decisions

**D1 — Real text over placeholder.** The chain only actually *chains* when
the real reasoning text is replayed (`reasoning_content` is concatenated into
the context for tool-scope requests). The placeholder exists solely to
satisfy the API contract without a blank-chain warning; it can silently
degrade to 0 reasoning deltas (measured live), which is why the native layer
(D1) matters more than the wire layer.

**D2 — `" "` not `""`.** LiteLLM and DeepSeek treat `""` as absent; a single
space is non-empty, satisfies validation, silences the gateway warning, and
is byte-for-byte the placeholder the gateway would inject anyway — so the
prefix cache is not perturbed more than it already would be.

**D3 — Signature stamping instead of payload rewriting.** Rebuilding the
serialized `reasoning_content` by hand at the wire layer would duplicate pi's
serializer logic and drift with pi releases. Stamping the native block
(single field) lets pi's own, version-maintained serializer do the replay.

**D4 — Strip `thinking: disabled` on tool-call continuations.** Open WebUI
sends the marker automatically on tool-call continuations, which is a silent
kill-switch. Pi sends it only when the user's thinking level is off. The
strip applies whenever the history is a tool-call continuation (assistant
`tool_calls` present), where a disabled marker is broken by definition for
DeepSeek (0 reasoning deltas, verified live); non-tool turns keep whatever
thinking setting the user chose. The continuation flag is evaluated from the
history **before** any forcing (single pass), so placeholder-only histories
(replay failed) still get the strip.

**D5 — Fail-open everywhere.** Every hook is a pure function with early
guards; if pi's payload shapes change, the extension does nothing rather than
crash a turn.

## 7. Scope detection

`isDeepSeekModel(provider, modelId, baseUrl)` matches:

- `provider === "deepseek"`,
- baseUrl containing `deepseek.com`,
- model id with the `deepseek/` or `deepseek-` prefix (covers LiteLLM-style
  gateways that keep the prefix),
- any extra prefix in `PI_DEEPSEEK_REASONING_EXTRA` (escape hatch).

*Known gap:* a gateway that strips the prefix (bare `deepseek-v4-flash`
under a custom provider) is silently missed — PLAN.md P2.

## 8. Validation evidence

### Offline

- 7 unit checks (`verify-extension.mjs`) against the **built artifact**
  (`dist/index.js`): hook registration, scope detection, signature stamping
  (idempotent), wire forcing (real text preserved, missing → `" "`),
  no-touch outside tool scope, thinking-strip, message_end stamping.

### Live (against `deepseek/deepseek-v4-flash` via LiteLLM)

A/B with three conditions × N repetitions on authentic tool-call
continuations:

- **No fix** — assistant rebuilt without `reasoning_content`.
- **Wire fix only** — placeholder `" "`.
- **Native + wire fix** — real text replayed.

Consistent findings: placeholder-only can drop continuation reasoning to
**0 deltas**; real-text replay never produced 0 and at its best showed 4×
more reasoning with visible chain continuity (`"Hemos recibido el resultado
de la herramienta... Basándome en el resultado..."`). `thinking: disabled`
always yielded **0 deltas** vs 17–38 without, across runs.

### Production observation

After the extension went live in a real pi session, the LiteLLM warning
`assistant message is missing reasoning_content...` (the exact symptom of the
broken contract) **stopped appearing** in the gateway logs.

## 9. Review findings and backlog

A full code review found **no I/O, timers, state, or concurrency choke
points** — the hooks are pure in-memory transforms. The backlog is
hardening, not redesign; tracked in `PLAN.md`:

| Priority | Item |
|----------|------|
| P0 | `thinking:disabled` strip semantics (evaluate continuation flag before forcing) |
| P1 | Single-pass wire normalization (three O(n) scans → one) |
| P1 | `Array.isArray` guards on `content` iteration |
| P2 | Scope detection for prefix-stripped gateways |
| P3 | Unit tests (strip semantics, robustness, determinism) + observability |

## 10. Security

- The extension performs no network access and never reads credentials.
- API keys are never stored in the repo; live probes read them from
  `$LITELLM_API_KEY` or a `0600` file; `.gitignore` blocks `.env`, `*.key`,
  `.npmrc`.
- The published tarball ships `dist/` only; `src/` stays in git (GitHub)
  and never reaches npm.

## 11. Packaging and versioning

- Repo root is the npm package: `src/` in git, `dist/` built and ignored.
- `files: ["dist"]` + `.npmignore` backstop → tarball contains only the
  built output, LICENSE, README, package.json (verified with
  `npm pack --dry-run`).
- `pi.extensions: ["./dist/index.js"]` declares the pi entry point.
- Semver via `npm version patch|minor|major` (commit + `vX.Y.Z` tag),
  bumps mapped from conventional commit types; `preversion` runs
  typecheck + verify; `prepublishOnly` rebuilds.

## 12. Appendix — DeepSeek API reference notes

- `reasoning_content` is returned at the same level as `content`
  (Chat Completions), also as stream deltas.
- `usage.completion_tokens_details.reasoning_tokens` accounts reasoning
  tokens.
- Thinking toggle: `{"thinking": {"type": "enabled|disabled"}}`; effort:
  `{"reasoning_effort": "low|high|max"}` (default enabled, effort `high`).
- Without `tools` in the request, passed-back `reasoning_content` is ignored
  and not concatenated — the contract only applies to tool scope.
