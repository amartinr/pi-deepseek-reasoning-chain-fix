# DeepSeek Reasoning Chain

Pi Coding Agent extension that keeps DeepSeek reasoning working across
tool-calling turns, plus a validation harness with live probes.

## What it does

- Preserves the real reasoning text on tool-call continuations.
- Guarantees DeepSeek's required `reasoning_content` field on every request
  in tool scope.
- Removes the `thinking: disabled` flag that stops reasoning mid-turn.

DeepSeek requires `reasoning_content` on every assistant message once a
history contains tool calls. Without this fix, reasoning degrades or stops
entirely after the first tool call.

## Install

```bash
pi install pi-deepseek-reasoning-chain-fix@0.1.0   # or: pi install ./extensions
```

Then run `/reload` in pi.

Works with any DeepSeek model, direct or behind a gateway (e.g. LiteLLM).

## Repository

| Path | Purpose |
|------|---------|
| `extensions/` | The pi extension (npm package: `src/` in git, `dist/` on npm) |
| `tests/`, `probes/`, `alg-sim/` | Validation harness (unit + live probes) |

## Validation

- 59 offline unit tests (forcing, replay, cleanup).
- Extension unit checks against the built artifact.
- Live A/B against a LiteLLM gateway.

## License

MIT
