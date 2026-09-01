# pi-deepseek-reasoning-chain

Pi extension that keeps DeepSeek reasoning working across tool-calling turns.

DeepSeek requires `reasoning_content` on every assistant message once the
history contains tool calls. This extension fixes outbound requests so the
field is always present and the real reasoning text survives between turns.
Without it, DeepSeek stops reasoning (blank chain) or errors after the first
tool call.

## Install

```bash
pi install pi-deepseek-reasoning-chain@0.1.0
# or from this directory:
pi install /work/extensions
# or manual copy (single-file source):
cp src/index.ts ~/.pi/agent/extensions/deepseek-reasoning-chain.ts
```

Then run `/reload` in pi.

Compatible with any DeepSeek model, direct or behind a gateway that keeps the
`deepseek/` prefix (e.g. LiteLLM).

## Configuration

Optional: `PI_DEEPSEEK_REASONING_EXTRA` — comma-separated model prefixes to
treat as DeepSeek.

## Development

```bash
npm install          # dev dependencies
npm run build        # src/ -> dist/
npm run verify       # unit checks against the built artifact
node verify-live.mjs # live A/B against a gateway
npm pack --dry-run   # inspect the published tarball (dist only)
```

## Versioning

Semver via `npm version patch|minor|major` (commit + `vX.Y.Z` tag).
Bumps follow conventional commit types: `fix:` → patch, `feat:` → minor,
breaking → major (pre-1.0: minor).

## License

MIT
