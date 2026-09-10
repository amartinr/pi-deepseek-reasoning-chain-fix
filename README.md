# pi-deepseek-reasoning-chain-fix

Pi extension that keeps DeepSeek reasoning working across tool-calling turns.

DeepSeek requires `reasoning_content` on every assistant message once the
history contains tool calls. This extension fixes outbound requests so the
field is always present and the real reasoning text survives between turns.
Without it, DeepSeek stops reasoning (blank chain) or errors after the first
tool call.

## Install

From npm:

```bash
pi install npm:@amartinr/pi-deepseek-reasoning-chain-fix@0.2.3
```

From GitHub:

```bash
pi install git:github.com/amartinr/pi-deepseek-reasoning-chain-fix@v0.2.3
```

Manual copy (single-file source, for development):

```bash
cp src/index.ts ~/.pi/agent/extensions/pi-deepseek-reasoning-chain-fix.ts
```

Then run `/reload` in pi.

Apply to the models you list in `config.json` — typical DeepSeek ids like
`deepseek/deepseek-v4-flash` (direct or behind a gateway such as LiteLLM)
work out of the box.

## Configuration

The extension applies only to the model ids listed in
`~/.pi/agent/extensions/pi-deepseek-reasoning-chain-fix/config.json`:

```json
{
  "models": [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro"
  ],
  "replayReasoning": true
}
```

- `models` — model ids the fix applies to; exact or prefix match
  (case-insensitive), so `"deepseek/"` covers every deepseek-routed model.
  An empty or missing list leaves the extension **inert** — the safe
  default.
- `replayReasoning` — `true` (default): the real reasoning text is replayed
  on continuations (chaining). `false`: compliance only — a `" "`
  placeholder is sent to satisfy DeepSeek's contract, without chaining the
  real text.

Override the config path with `PI_DEEPSEEK_REASONING_CONFIG`. The load log
reports the active ids and the mode.

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
