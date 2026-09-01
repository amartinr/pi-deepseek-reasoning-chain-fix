"""
title: Agent Loop Guard
id: agent_loop_guard
author: A. Martin
author_url: https://github.com/amartinr
git_url: https://github.com/amartinr/open-webui-extensions.git
description: Pipe function that prevents AI agents from entering infinite tool-calling loops, without wasting tool results or burning LLM tokens. For DeepSeek-class models it also forces reasoning_content on assistant messages of tool-calling histories (required by the DeepSeek API contract, missing field silently degrades multi-turn reasoning). Opt-in per-request diagnostics behind the DEBUG_LOG valve.
required_open_webui_version: 0.5.0
requirements: httpx, pydantic
version: 2.17.5
licence: MIT
"""

from pydantic import BaseModel, Field, model_validator
from typing import AsyncGenerator, Awaitable, Callable, Optional
import httpx
import json
import logging
import re
import time

log = logging.getLogger(__name__)


GUARD_MARKER = "[Tool call budget exhausted]"


_REASONING_REPLAY_PATCHED = False
_LAST_PATCH_WARN = 0.0


def _rate_limited_warning(level: str, msg: str, exc: str = "") -> bool:
    """Emit a log line at most once per 5 minutes (module-wide).

    Returns True when the line was actually emitted.
    """
    global _LAST_PATCH_WARN
    now = time.monotonic()
    if now - _LAST_PATCH_WARN < 300.0:
        return False
    _LAST_PATCH_WARN = now
    (log.warning if level == "warning" else log.info)(msg, exc)
    return True


def _install_reasoning_replay_patch() -> bool:
    """Monkey-patch Open WebUI's get_reasoning_format so pipe models replay the
    REAL reasoning text as `reasoning_content` on tool-call continuations.

    Open WebUI's get_reasoning_format() returns None for every OpenAI-compatible
    model (owned_by='openai'), so convert_output_to_messages() DISCARDS the
    stored reasoning when it rebuilds assistant history for a tool-call
    continuation (verified against the open-webui source: with
    reasoning_format=None the reasoning block is skipped). The replayed
    assistant then reaches the gateway without reasoning_content and the pipe
    can only force a single-space placeholder. Patching get_reasoning_format to
    return 'reasoning_content' for pipe models makes Open WebUI rebuild the
    assistant with the real text (reasoning_format='reasoning_content' ->
    pending_reasoning -> message['reasoning_content']).

    A/B probe against LiteLLM (probes/litellm/03_replay_ab.py): both variants
    reason on every continuation (8/8), but with the real text the continuation
    reasoning is ~19% richer (avg 91 vs 76 chars) and continues the previous
    chain instead of re-deriving from scratch.

    Scoped to pipe models only (owned_by='openai' AND a 'pipe' key), so direct
    OpenAI connections and Ollama/llama.cpp models keep their original
    behavior. Fails open: if Open WebUI changes these internals the patch
    raises and the pipe falls back to placeholder forcing (no crash).
    Idempotent via a module marker.

    Returns True when the patch is (or was) active.
    """
    global _REASONING_REPLAY_PATCHED
    if _REASONING_REPLAY_PATCHED:
        return True
    try:
        from open_webui.utils import middleware as _mw

        current = _mw.get_reasoning_format
        if getattr(current, "_alg_reasoning_patched", False):
            _REASONING_REPLAY_PATCHED = True
            return True

        original = current

        def _patched(model):
            result = original(model)
            if result is not None:
                return result
            if (
                isinstance(model, dict)
                and model.get("owned_by") == "openai"
                and model.get("pipe")
            ):
                return "reasoning_content"
            return result

        _patched._alg_reasoning_patched = True
        _mw.get_reasoning_format = _patched
        _REASONING_REPLAY_PATCHED = True
        log.info(
            "agent-loop-guard: reasoning replay patch installed "
            "(REPLAY_REASONING_TEXT valve on)"
        )
    except Exception as exc:
        # Rate-limited: with the valve on and the failure persistent, this
        # would otherwise fire on every request.
        _rate_limited_warning(
            "warning",
            "agent-loop-guard: could not install reasoning replay patch "
            "(falling back to placeholder forcing): %s",
            str(exc),
        )
    return _REASONING_REPLAY_PATCHED


# --------------------------------------------------------------------------
# Message templates (single source of truth)
# --------------------------------------------------------------------------

MSG_TOOL_LOOP = (
    "{marker} - loop detected\n"
    "{tool}: {total} identical calls exceed the limit.\n"
    "Stop repeating. Try a different tool or summarise what you have."
)

MSG_TOOL_RUNAWAY = (
    "{marker} - turn limit reached\n"
    "You've used all {max_calls} allowed calls this turn (attempted {total}).\n"
    "No more tools available. Summarise what you have."
)

MSG_NOTIFY_LOOP = "\U0001f527 {tool} budget exhausted after too many identical calls."
MSG_NOTIFY_RUNAWAY = "\U0001f527 Tool call budget exhausted ({total}/{max_calls})."
MSG_COUNTER = "\U0001f527 Remaining tool calls: {remaining}/{max_calls}"


def _build_guard_message(
    status: str, tool: str | None, total: int, max_calls: int
) -> str:
    """Build the text that replaces the tool result when budget is exhausted."""
    if status == "loop":
        return MSG_TOOL_LOOP.format(marker=GUARD_MARKER, tool=tool, total=total)
    elif status == "runaway":
        return MSG_TOOL_RUNAWAY.format(
            marker=GUARD_MARKER, total=total, max_calls=max_calls
        )
    return ""


# --------------------------------------------------------------------------
# DeepSeek reasoning forcing (transport-independent)
# --------------------------------------------------------------------------
#
# DeepSeek requires `reasoning_content` on every assistant message of a
# tool-calling history; a missing field silently degrades multi-turn reasoning.
# This is the DeepSeek API contract — it applies through ANY OpenAI-compatible
# gateway. LiteLLM itself warns about it (transformation.py): when the field is
# missing it injects a blank placeholder and the model receives an empty
# reasoning chain. Open WebUI rebuilds assistant messages from stored output
# items and, for OpenAI-compatible providers (owned_by='openai', which includes
# LiteLLM and Bifrost), omits `reasoning_content` on tool-call continuations.
# Filter inlets do not run on tool-call continuations, so this pipe — the
# single choke point for every outbound request to the gateway — forces the
# field (a single space is enough for DeepSeek to keep reasoning — and it
# matches the placeholder LiteLLM would inject anyway).


def _messages_summary(messages: list, verbose: bool = False) -> str:
    """Compact per-message shape summary for diagnostics.

    One token per message: role + flags — T (has tool_calls), R (has
    reasoning_content), c (content is a list).

    With `verbose=True` the reasoning flag distinguishes R0 (present-but-
    empty — the dangerous case for DeepSeek: an empty reasoning chain is
    replayed to the model) from R<n> (present with text of length n).
    """
    parts = []
    for m in messages if isinstance(messages, list) else []:
        if not isinstance(m, dict):
            parts.append("?")
            continue
        role = m.get("role", "?")
        flags = ""
        if isinstance(m.get("tool_calls"), list) and len(m["tool_calls"]) > 0:
            flags += "T"
        rc = m.get("reasoning_content")
        if isinstance(rc, str):
            if verbose:
                flags += "R0" if rc == "" else f"R{len(rc)}"
            else:
                flags += "R"
        if isinstance(m.get("content"), list):
            flags += "c"
        parts.append(f"{role}{flags or '-'}")
    return "[" + " ".join(parts) + "]"


def _history_has_tool_calls(messages: list) -> bool:
    """True when the history contains an assistant message with tool calls.

    Once a tool call has happened, DeepSeek requires `reasoning_content` on
    every assistant message of every subsequent request — regardless of
    whether that request still ships `tools`. Tool-call continuations from
    Open WebUI bypass filter inlets, so this pipe is the single choke point
    that sees every outbound request to the gateway.
    """
    return any(
        isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and isinstance(msg.get("tool_calls"), list)
        and len(msg["tool_calls"]) > 0
        for msg in messages
    )


def _force_reasoning_content_on_assistant(messages: list) -> int:
    """Guarantee every assistant message carries a non-empty `reasoning_content`.

    DeepSeek silently degrades multi-turn reasoning when a tool-calling
    history replays an assistant message without `reasoning_content`.
    Open WebUI rebuilds assistant messages without the field on tool-call
    continuations, and LiteLLM treats a MISSING or EMPTY ("") value as
    absent: it injects a single-space placeholder and warns about it
    (transformation.py, "DeepSeek thinking mode"). Forcing a single space
    is exactly what LiteLLM would inject anyway, so the placeholder is
    explicit and the warning is silenced.

    Returns the number of assistant messages that were forced (diagnostics).
    """
    forced = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            if not isinstance(msg.get("reasoning_content"), str) or not msg["reasoning_content"]:
                msg["reasoning_content"] = " "
                forced += 1
    return forced


def _force_reasoning_on_gateway_payload(body: dict) -> int:
    """Force `reasoning_content` on assistant messages once tool-calling is in
    scope (request `tools` or tool-call history). DeepSeek contract fix,
    transport-independent (validated against LiteLLM + Bifrost).

    Never touches user/system/tool messages and is deterministic, so the
    provider prefix cache is not invalidated by the rewrite.

    Returns the number of assistant messages that were forced (diagnostics).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    has_tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    if not (has_tools or _history_has_tool_calls(messages)):
        return 0
    return _force_reasoning_content_on_assistant(messages)


def _normalize_thinking_for_gateway(body: dict) -> bool:
    """Strip Open WebUI's thinking:disabled on tool-call continuations.

    Open WebUI sends thinking={'type': 'disabled'} (and drops reasoning_effort)
    on server-side tool-call continuations. For DeepSeek that is a hard
    kill-switch: it disables reasoning entirely (0 reasoning deltas), and a
    demanding system prompt cannot override it. The user's own turns never
    send 'disabled' — they send 'enabled' or omit the field — so removing the
    disabled marker restores DeepSeek's default thinking and reasoning resumes
    (verified: no thinking field + no effort still reasons).

    Returns True when the field was removed.
    """
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        body.pop("thinking", None)
        return True
    return False


# --------------------------------------------------------------------------
# Attached-files cleanup (cache-safe)
# --------------------------------------------------------------------------
#
# Open WebUI injects <attached_files> blocks in two places:
#   - the image_filter inlet prepends ONE block to the LAST user message
#     with the CURRENT turn's images only (since filter v2.12.0; before
#     that it was the union of all conversation images, re-announced
#     every turn);
#   - the core's add_file_context() prepends one block per stored user
#     message that has files (that message's own files, relative URLs).
#
# So the payload can carry several per-message core blocks plus the
# filter's current-turn block, with the same file tagged in more than
# one place. This cleanup collapses duplicate tags WITHIN each user
# message — a simple UUID match (the same upload is tagged by both the
# core and the filter in the current turn; the file UUID is unique) plus
# a content-hash match (two *different* UUIDs with identical bytes
# collapse too) — and re-emits image tags in our canonical format (id +
# absolute URL). Multiple blocks in one message collapse into one.
#
# The dedup scope is PER USER MESSAGE (one message = one turn), never
# across messages: a file deliberately re-uploaded in a later turn gets
# a NEW block in that turn and stays visible to the agent — it is NOT
# deduplicated away. Historical per-message blocks stay byte-stable
# between turns (the cleanup is a deterministic pure function of each
# message + the stored history), so the cached prefix extends through
# the whole history and nothing is re-presented from history. (The
# cross-message dedup that earlier versions did was only needed for the
# pre-v2.12.0 filter, which re-announced a moving union block every
# turn; the current filter never touches historical messages, so
# cross-message dedup's only remaining effect was hiding deliberate
# re-uploads — removed in v2.4.0.)
#
# Fail-open by design: callers wrap the call in try/except and forward
# the payload unchanged on error.

_BLOCK_RE = re.compile(r"<attached_files>(.*?)</attached_files>", re.DOTALL)
_FILE_TAG_RE = re.compile(r"<file\b([^>]*?)/?>")
_ATTR_RE = re.compile(r'([\w:.-]+)="([^"]*)"')
_FILES_URL_RE = re.compile(r"/api/v1/files/([^/]+)/content")
_PLACEHOLDER = "(base64 stripped)"


def _parse_file_tags(text: str) -> list[dict]:
    """Parse all `<file .../>` tags inside `<attached_files>` blocks in `text`.

    Returns a list of tag dicts: `{"raw": str, "attrs": [(name, value), ...]}`
    with attribute order preserved. Blocks that contain no `<file>` tag are
    ignored (defensive: user text that literally contains `<attached_files>`
    is left untouched).
    """
    tags = []
    for block in _BLOCK_RE.finditer(text or ""):
        for raw in _FILE_TAG_RE.findall(block.group(1)):
            attrs = [(m.group(1), m.group(2)) for m in _ATTR_RE.finditer(raw)]
            if not attrs:
                continue
            tags.append({"raw": f"<file{raw}>", "attrs": attrs})
    return tags


def _count_blocks_with_file_tags(text: str) -> int:
    """Number of `<attached_files>` blocks in `text` that contain a `<file>` tag.

    Blocks without any `<file>` tag are ignored by the parser (defensive),
    so they are not counted as collapsible blocks.
    """
    count = 0
    for block in _BLOCK_RE.finditer(text or ""):
        if _FILE_TAG_RE.search(block.group(1)):
            count += 1
    return count


def _is_image_attrs(attrs) -> bool:
    """True when the tag describes an image (by `type` or `content_type`).

    Accepts either the raw attrs list `[(name, value), ...]` or a dict.
    """
    if isinstance(attrs, list):
        attrs = dict(attrs)
    if attrs.get("type") == "image":
        return True
    return (attrs.get("content_type") or "").lower().startswith("image/")


def _extract_uuid(tag: dict) -> str:
    """Extract the file UUID from a parsed `<file>` tag.

    The UUID can appear as the `id` attribute, as a bare-UUID `url` (this
    deployment's raw upload form: `<file type="file" url="{uuid}" .../>`),
    or inside a `/api/v1/files/{uuid}/content` URL (relative or absolute).
    Returns "" when the tag carries no UUID (external URLs, `data:` URIs,
    placeholder tags).
    """
    attrs = dict(tag["attrs"])
    url = attrs.get("url", "") or ""
    if url and url != _PLACEHOLDER:
        m = _FILES_URL_RE.search(url)
        if m:
            return m.group(1)
        if "/" not in url and ":" not in url:
            return url  # bare UUID (raw upload form)
    file_id = attrs.get("id", "") or ""
    if file_id and file_id != _PLACEHOLDER:
        return file_id
    return ""


def _canonical_type(attrs: dict) -> str:
    """Normalize the tag's `type` attribute: images are always `image`."""
    ct = (attrs.get("content_type") or "").lower()
    if attrs.get("type") == "image" or ct.startswith("image/"):
        return "image"
    return attrs.get("type") or "file"


def _file_dedup_key(tag: dict) -> str:
    """Canonical key for a parsed `<file>` tag.

    Only IMAGE tags take part in the cleanup; non-image tags return "" so
    they are always kept and never participate in dedup. The file UUID is
    unique, so dedup is a simple UUID match: the same image collapses
    across the filter's absolute form, the core's relative form, and this
    deployment's bare-UUID raw form. External image URLs (no UUID) key by
    the full URL. Placeholder tags (`(base64 stripped)`) return "" — they
    are never deduplicated either.
    """
    attrs = dict(tag["attrs"])
    if not _is_image_attrs(attrs):
        return ""  # non-image tags: never deduplicated, never seen-marked
    uuid = _extract_uuid(tag)
    if uuid:
        return f"id:{uuid}"
    url = attrs.get("url", "") or ""
    if url and url != _PLACEHOLDER:
        return f"url:{url}"
    return ""


def _normalize_tag(tag: dict, base_url: str) -> str:
    """Re-emit a tag in the pipe's canonical format.

    Only IMAGE tags are re-emitted in our canonical format — `type="image"`,
    `id="{uuid}"`, and an absolute `/api/v1/files/{uuid}/content` URL
    (relative when no base URL is available) — plus `content_type` and
    `name` when present. The same image therefore renders identically
    regardless of which source produced it, keeping the builtin `view_file`
    tool (uses `id`) and external URL loaders (ComfyUI) working.

    Non-image tags (PDFs, documents, external URLs, `data:` URIs,
    placeholders) are re-emitted **unchanged**, with relative URLs prefixed
    when a base URL exists (attribute order preserved, deterministic).
    """
    attrs = tag["attrs"]
    if _is_image_attrs(attrs):
        uuid = _extract_uuid(tag)
        if uuid:
            attrs_dict = dict(attrs)
            parts = [f'type="{_canonical_type(attrs_dict)}"', f'id="{uuid}"']
            rel = f"/api/v1/files/{uuid}/content"
            parts.append(f'url="{base_url + rel if base_url else rel}"')
            for name in ("content_type", "name"):
                value = attrs_dict.get(name)
                if value:
                    parts.append(f'{name}="{value}"')
            return "<file " + " ".join(parts) + "/>"

    out = []
    for name, value in attrs:
        if name == "url" and value.startswith("/") and base_url:
            value = f"{base_url}{value}"
        out.append(f'{name}="{value}"')
    return "<file " + " ".join(out) + "/>"


def _build_block(tags: list[dict], base_url: str) -> str:
    """Build a single `<attached_files>` block (core's exact format).

    Image tags are re-emitted in our canonical format; non-image tags are
    re-emitted unchanged (preserving their original relative URL)."""
    if not tags:
        return ""
    lines = [_normalize_tag(t, base_url) for t in tags]
    return "<attached_files>\n" + "\n".join(lines) + "\n</attached_files>\n\n"


def _dedupe_tags(
    tags: list[dict], seen: set[str], hash_lookup: dict[str, str] | None = None
) -> list[dict]:
    """Keep tags whose canonical key has not been seen before.

    The `seen` set is scoped by the caller to ONE user message (per-turn
    dedup): it collapses the filter's current-turn block with the core's
    block of the same message, and never removes a deliberate re-upload
    from a later turn.

    Placeholder tags (key "") are always kept and never added to `seen`.
    Each dropped duplicate is logged at info level for debuggability.

    `hash_lookup` maps a file UUID to its stored sha256
    (`meta["file_hash"]`) — a content-level dedup backstop. The image_filter
    is supposed to converge every source on the CURRENT upload's UUID, but
    if it fails (e.g. hash metadata unavailable, re-encoded base64 copy)
    two DIFFERENT UUIDs can reference the same bytes. The pipe is the last
    code before the provider, so it also marks `hash:{digest}` in `seen`:
    a second tag with different UUID but identical content collapses too
    (first occurrence wins, same as the UUID rule).
    """
    kept = []
    for tag in tags:
        key = _file_dedup_key(tag)
        if key:
            if key in seen:
                log.info(
                    "attached_files: dropping duplicate tag %s "
                    "(already tagged earlier in this request)",
                    key,
                )
                continue
            seen.add(key)
            if key.startswith("id:") and hash_lookup:
                digest = hash_lookup.get(key[3:])
                if digest:
                    hkey = f"hash:{digest}"
                    if hkey in seen:
                        log.info(
                            "attached_files: dropping tag %s — same content as "
                            "an earlier tag (sha256=%s...)",
                            key,
                            digest[:12],
                        )
                        continue
                    seen.add(hkey)
        kept.append(tag)
    return kept


def _collect_image_uuids(messages: list) -> list[str]:
    """Collect the file UUIDs of every image tag in every `<attached_files>`
    block (deduplicated), so the pipe can resolve their content hashes
    before the cleanup runs. Non-image tags are skipped — they never take
    part in the cleanup."""
    uuids: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        parts = (
            [{"type": "text", "text": content}]
            if isinstance(content, str)
            else (
                [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
                if isinstance(content, list)
                else []
            )
        )
        for part in parts:
            for tag in _parse_file_tags(part.get("text", "")):
                if not _is_image_attrs(tag["attrs"]):
                    continue
                uuid = _extract_uuid(tag)
                if uuid and uuid not in seen:
                    seen.add(uuid)
                    uuids.append(uuid)
    return uuids


async def _resolve_content_hashes(uuids: list[str]) -> dict[str, str]:
    """Resolve file UUIDs to their stored sha256 (`meta["file_hash"]`).

    Best-effort and fail-open: any lookup error skips that UUID (the
    cleanup then falls back to UUID-only dedup). Only image UUIDs reach
    this point, and each UUID resolves once per request."""
    resolved: dict[str, str] = {}
    for uuid in uuids:
        try:
            from open_webui.models.files import Files

            fobj = await Files.get_file_by_id(uuid)
            digest = (getattr(fobj, "meta", None) or {}).get("file_hash")
            if digest:
                resolved[uuid] = digest
        except Exception:
            continue
    return resolved


def _cleanup_attached_files(
    messages: list, base_url: str = "", hash_lookup: dict[str, str] | None = None
) -> dict:
    """Collapse and deduplicate `<attached_files>` blocks WITHIN each user message.

    Pure function of the payload (deterministic → cache-safe). Dedup is
    scoped PER USER MESSAGE (one message = one turn): the same upload
    tagged by both the core's `add_file_context()` and the image_filter in
    the current turn collapses to one tag, and two different UUIDs with
    identical content (`hash_lookup`) collapse too. Files are NEVER
    deduplicated across messages — a deliberate re-upload in a later turn
    keeps its own block in that turn and stays visible to the agent.
    Multiple blocks in one message collapse into one; relative URLs are
    normalized to absolute when `base_url` is available. Mutates `messages`
    in place (the same dicts survive the middleware's shallow copies).
    Fail-open by design: callers wrap in try/except and forward unchanged.

    Returns a stats dict (for logging):
    `{"user_messages", "blocks_found", "blocks_kept", "tags_kept",
    "tags_dropped", "image_tags_kept"}`.
    """
    stats = {
        "user_messages": 0,
        "blocks_found": 0,
        "blocks_kept": 0,
        "tags_kept": 0,
        "tags_dropped": 0,
        "image_tags_kept": 0,
    }

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        stats["user_messages"] += 1

        # Dedup scope: THIS message only. One user message = one turn; a
        # file re-uploaded in a later turn must stay visible in its own
        # block. The image_filter only announces the current turn, so the
        # core's per-message blocks are the only historical source and are
        # already stable — a fresh `seen` per message keeps that stability.
        seen: set[str] = set()

        content = message.get("content")
        if isinstance(content, str):
            blocks_found = _count_blocks_with_file_tags(content)
            tags = _parse_file_tags(content)
            if not tags:
                continue
            kept = _dedupe_tags(tags, seen, hash_lookup)
            dropped = len(tags) - len(kept)
            stats["blocks_found"] += blocks_found
            stats["tags_kept"] += len(kept)
            stats["tags_dropped"] += dropped
            stats["image_tags_kept"] += sum(
                1 for t in kept if _is_image_attrs(t["attrs"])
            )

            cleaned = re.sub(r"^\n+", "", _BLOCK_RE.sub("", content))
            block = _build_block(kept, base_url)
            if block:
                message["content"] = block + cleaned
            elif cleaned:
                message["content"] = cleaned
            else:
                message["content"] = [{"type": "text", "text": ""}]
            blocks_kept = 1 if block else 0
            stats["blocks_kept"] += blocks_kept
            log.info(
                "attached_files: user message %d/%d — found %d block(s), "
                "%d tag(s); kept %d (%d dropped as duplicates); "
                "collapsed to %d block(s)",
                stats["user_messages"],
                len(messages),
                blocks_found,
                len(tags),
                len(kept),
                dropped,
                blocks_kept,
            )

        elif isinstance(content, list):
            tags = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    tags.extend(_parse_file_tags(part.get("text", "")))
            if not tags:
                continue
            blocks_found = sum(
                _count_blocks_with_file_tags(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            kept = _dedupe_tags(tags, seen, hash_lookup)
            dropped = len(tags) - len(kept)
            stats["blocks_found"] += blocks_found
            stats["tags_kept"] += len(kept)
            stats["tags_dropped"] += dropped
            stats["image_tags_kept"] += sum(
                1 for t in kept if _is_image_attrs(t["attrs"])
            )

            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    cleaned = re.sub(
                        r"^\n+", "", _BLOCK_RE.sub("", part.get("text", ""))
                    )
                    if cleaned:
                        new_content.append({**part, "text": cleaned})
                else:
                    new_content.append(part)

            block = _build_block(kept, base_url)
            if block:
                new_content.insert(0, {"type": "text", "text": block})
            message["content"] = (
                new_content if new_content else [{"type": "text", "text": ""}]
            )
            blocks_kept = 1 if block else 0
            stats["blocks_kept"] += blocks_kept
            log.info(
                "attached_files: user message %d/%d — found %d block(s), "
                "%d tag(s); kept %d (%d dropped as duplicates); "
                "collapsed to %d block(s)",
                stats["user_messages"],
                len(messages),
                blocks_found,
                len(tags),
                len(kept),
                dropped,
                blocks_kept,
            )

    log.info(
        "attached_files: cleanup done — %d user message(s) scanned, "
        "blocks %d → %d, image tag(s) kept %d, duplicate(s) dropped %d "
        "(base_url=%s)",
        stats["user_messages"],
        stats["blocks_found"],
        stats["blocks_kept"],
        stats["image_tags_kept"],
        stats["tags_dropped"],
        base_url or "none",
    )
    return stats


class Pipe:
    class Valves(BaseModel):
        GATEWAY_BASE_URL: str = Field(
            default="",
            description="Base URL for the OpenAI-compatible gateway (e.g. LiteLLM).",
        )
        GATEWAY_AUTH_HEADER: str = Field(
            default="Authorization",
            description="HTTP header name for the API key (e.g. 'Authorization', 'x-api-key').",
        )
        GATEWAY_AUTH_VALUE: str = Field(
            default="",
            description="Credential value sent in the configured auth header (e.g. 'Bearer sk-...').",
            json_schema_extra={"input": {"type": "password"}},
        )
        GATEWAY_CUSTOM_HEADERS: str = Field(
            default="",
            description="JSON object of extra HTTP headers to send with every gateway request. "
            'Example: {"x-tenant-id": "myhost", "x-trace-id": "abc"}. '
            "Leave empty if not needed.",
        )
        MAX_TOOL_CALLS_PER_TURN: int = Field(
            default=15,
            description="Max tool calls in a turn before the guard fires. Set to 0 to disable.",
        )
        MAX_CONSECUTIVE_TOOL_CALLS: int = Field(
            default=4,
            ge=3,
            description="Max consecutive identical tool calls before budget is exhausted (min 3).",
        )
        TOOL_BLOCKLIST: str = Field(
            default="",
            description="Comma-separated (or newline-separated) tool names to REMOVE from the agent's tool list. "
            'Example: "delete_file, terminal_execute".',
        )
        ATTACHED_FILES_CLEANUP: bool = Field(
            default=True,
            description="Collapse and deduplicate <attached_files> blocks WITHIN each user message "
            "(per-turn): the core's add_file_context() block and the image_filter's "
            "current-turn block for the same upload collapse to one tag. Re-uploads in "
            "later turns keep their own block (never deduplicated away), and historical "
            "messages stay byte-stable between turns, so the prefix cache is preserved. "
            "Set to False to forward payloads unchanged.",
        )
        DEBUG_LOG: bool = Field(
            default=False,
            description="Per-request diagnostics: messages summary with R0 (present-but-empty) vs "
            "R<n> (text length) reasoning flags, last assistant reasoning_content "
            "length/empty/prefix, and the full outbound request (url/headers/payload). "
            "Off by default — verbose, only needed when debugging reasoning behavior.",
        )
        REPLAY_REASONING_TEXT: bool = Field(
            default=True,
            description="Replay the REAL reasoning text on tool-call continuations by monkey-patching "
            "Open WebUI's get_reasoning_format for pipe models. Without it, Open WebUI "
            "discards the reasoning when rebuilding assistant history and the pipe can "
            "only force a single-space placeholder. A/B probe: continuation reasoning is "
            "~19% richer with the real text. Fragile: depends on Open WebUI internals; "
            "fails open (falls back to placeholder forcing). On by default; disable only "
            "if you prefer placeholder forcing over the patch.",
        )

        @model_validator(mode="after")
        def _check_runaway_gt_loop(self):
            """Ensure MAX_TOOL_CALLS_PER_TURN > MAX_CONSECUTIVE_TOOL_CALLS
            when both are enabled."""
            runaway = self.MAX_TOOL_CALLS_PER_TURN
            loop = self.MAX_CONSECUTIVE_TOOL_CALLS
            if runaway > 0 and loop >= runaway:
                raise ValueError(
                    f"MAX_TOOL_CALLS_PER_TURN ({runaway}) must be greater than "
                    f"MAX_CONSECUTIVE_TOOL_CALLS ({loop})."
                )
            return self

    class UserValves(BaseModel):
        MAX_TOOL_CALLS_PER_TURN: int = Field(
            default=0,
            ge=0,
            description="Max tool calls in a turn. 0 = use admin default.",
        )
        MAX_CONSECUTIVE_TOOL_CALLS: int = Field(
            default=0,
            ge=0,
            description="Max consecutive identical tool calls. 0 = use admin default.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._admin_valves = self.Valves()
        self._models_cache: list[dict] = []
        # Shared connection pool: reused across requests and tool-call
        # iterations (no fresh DNS+TLS handshake per call). httpx reuses
        # keep-alive connections and detects stale ones. Timeouts are set per
        # request; the client default is the non-stream budget.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    # ------------------------------------------------------------------
    # Model discovery (manifold)
    # ------------------------------------------------------------------

    async def pipes(self) -> list[dict]:
        if not self.valves.GATEWAY_BASE_URL:
            return [{"id": "config", "name": "⚠️ Configure gateway URL"}]

        headers = self._build_gateway_headers()
        url = f"{self.valves.GATEWAY_BASE_URL.rstrip('/')}/models"

        try:
            r = await self._client.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("Gateway unreachable during model discovery: %s", e)
            return self._models_cache or [
                {"id": "error", "name": "⚠️ Gateway unreachable"}
            ]

        self._models_cache = [
            {"id": m["id"], "name": f"🔧 {m.get('name', m['id'])}"}
            for m in data.get("data", [])
        ]
        log.info("Model discovery: %d models cached", len(self._models_cache))
        return self._models_cache

    # ------------------------------------------------------------------
    # Gateway helpers
    # ------------------------------------------------------------------

    async def _get_public_base_url(self, request) -> str:
        """Return the public base URL of this Open WebUI instance.

        Prefers the admin-configured "WebUI URL" setting (webui.url),
        falling back to the request's base URL when unset. Mirrors
        image_filter so normalized file URLs stay consistent. Returns ""
        when neither is available (relative URLs are then left as-is).
        """
        try:
            from open_webui.models.config import Config

            webui_url = await Config.get("webui.url")
            if webui_url:
                return str(webui_url).rstrip("/")
        except Exception as exc:
            log.warning("Failed to read webui.url config: %s", exc)

        base = getattr(request, "base_url", None)
        return str(base).rstrip("/") if base else ""

    def _build_gateway_headers(
        self,
        user: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        headers = {}
        if self.valves.GATEWAY_AUTH_VALUE:
            headers[self.valves.GATEWAY_AUTH_HEADER] = self.valves.GATEWAY_AUTH_VALUE
        else:
            log.warning("GATEWAY_AUTH_VALUE is empty.")

        if self.valves.GATEWAY_CUSTOM_HEADERS:
            try:
                raw_headers = json.loads(self.valves.GATEWAY_CUSTOM_HEADERS)
                if isinstance(raw_headers, dict):
                    user = user or {}
                    meta = metadata or {}
                    template_vars = {
                        "{{USER_ID}}": str(user.get("id", "") or ""),
                        "{{USER_NAME}}": str(user.get("name", "") or ""),
                        "{{USER_EMAIL}}": str(user.get("email", "") or ""),
                        "{{USER_ROLE}}": str(user.get("role", "") or ""),
                        "{{CHAT_ID}}": str(meta.get("chat_id", "") or ""),
                        "{{MESSAGE_ID}}": str(meta.get("message_id", "") or ""),
                    }
                    for k, v in raw_headers.items():
                        if not k:
                            continue
                        val = str(v) if v is not None else ""
                        for token, resolved in template_vars.items():
                            val = val.replace(token, resolved)
                        headers[k] = val
                else:
                    log.warning("GATEWAY_CUSTOM_HEADERS is not a JSON object")
            except json.JSONDecodeError as e:
                log.warning("GATEWAY_CUSTOM_HEADERS is not valid JSON: %s", e)

        return headers

    # ------------------------------------------------------------------
    # Tool blocklist helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_list(raw: str) -> set[str]:
        if not raw or not raw.strip():
            return set()
        return {t.strip() for t in re.split(r"[,\n\r]+", raw) if t.strip()}

    def _apply_tool_blocklist(self, body: dict) -> None:
        raw = getattr(self.valves, "TOOL_BLOCKLIST", "")
        if not raw or not raw.strip():
            return
        tools = body.get("tools", [])
        if not tools:
            return
        blocked = self._parse_tool_list(raw)
        actual_names = {
            t.get("function", {}).get("name") for t in tools if t.get("function", {})
        }
        unknown = blocked - actual_names
        if unknown:
            log.warning("TOOL_BLOCKLIST contains unknown names: %s", sorted(unknown))
        body["tools"][:] = [
            t for t in tools if t.get("function", {}).get("name") not in blocked
        ]
        tool_choice = body.get("tool_choice")
        if isinstance(tool_choice, str) and tool_choice in blocked:
            body.pop("tool_choice", None)

    # ------------------------------------------------------------------
    # Valve resolution
    # ------------------------------------------------------------------

    def _resolve_limit(self, user_val: int, admin_val: int) -> int:
        return user_val if user_val > 0 else admin_val

    # ------------------------------------------------------------------
    # Tool-call analysis
    # ------------------------------------------------------------------

    def _analyse(self, body: dict) -> tuple[bool, str | None, str, int, int]:
        """Analyse tool calls and decide if the guard should fire.

        Returns (should_block, tool_to_blame, block_kind, total, max_calls).

        block_kind is 'loop' or 'runaway'.  When should_block is False
        the other return values are meaningless.
        """
        messages = body.get("messages", [])
        max_calls = self._resolve_limit(
            self.valves.MAX_TOOL_CALLS_PER_TURN,
            self._admin_valves.MAX_TOOL_CALLS_PER_TURN,
        )
        max_consecutive = self._resolve_limit(
            self.valves.MAX_CONSECUTIVE_TOOL_CALLS,
            self._admin_valves.MAX_CONSECUTIVE_TOOL_CALLS,
        )

        # Extract real tool calls (skip those whose result was replaced by the guard)
        history: list[dict] = []

        # First pass: identify guard-replaced results
        guarded_ids: set[str] = set()
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and GUARD_MARKER in content:
                    guarded_ids.add(msg.get("tool_call_id", ""))

        # Second pass: collect real calls, skipping guarded ones
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("id", "") in guarded_ids:
                        continue
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    history.append({"name": tc["function"]["name"], "args": args})
        history.reverse()

        total = len(history)

        # Count consecutive identical calls
        consecutive = 0
        bad_tool = None
        if history:
            last_call = history[-1]
            for tc in reversed(history):
                if tc["name"] == last_call["name"] and tc["args"] == last_call["args"]:
                    consecutive += 1
                else:
                    break
            if consecutive >= 2:
                bad_tool = last_call["name"]

        # Loop detection: consecutive >= max_consecutive
        if max_consecutive > 0 and consecutive >= max_consecutive and bad_tool:
            return True, bad_tool, "loop", total, max_calls

        # Runaway: total >= max_calls (only if no loop)
        if max_calls > 0 and total >= max_calls:
            return True, None, "runaway", total, max_calls

        return False, None, "", total, max_calls

    # ------------------------------------------------------------------
    # Gateway proxy
    # ------------------------------------------------------------------

    async def _stream(self, payload: dict, headers: dict, url: str) -> AsyncGenerator[str, None]:
        # Red safety net: no data for 5 minutes ends the stream instead of
        # hanging forever (a stuck gateway would otherwise hold the request
        # open indefinitely). Long legitimate streams (thousands of events)
        # take seconds, so read=300 does not clip them.
        stream_timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
        async with self._client.stream(
            "POST", url, json=payload, headers=headers, timeout=stream_timeout
        ) as r:
            r.raise_for_status()
            stats = {"events": 0, "reasoning": 0, "content": 0, "tool_calls": 0, "done": False}
            async for line in r.aiter_lines():
                # Only forward well-formed OpenAI SSE data events. The rest is
                # discarded so it can never reach the OpenAI-compatible
                # consumer (Open WebUI's pipe handler turns any non-"data:"
                # line into chat CONTENT, and it emits its own closing
                # [DONE]/finish chunk). Concretely we keep:
                #   - "data: {...}" chunk events
                # and skip:
                #   - blank lines
                #   - SSE comments / keep-alives (": ..."), which a proxy or
                #     long-thinking gateway may inject; passing them through
                #     would corrupt the reasoning delta stream
                #   - "data: [DONE]" and any other noise
                if not isinstance(line, str):
                    continue
                stripped = line.strip()
                if not stripped.startswith("data:") or stripped == "data: [DONE]":
                    continue
                body = stripped.removeprefix("data:").strip()
                if not body.startswith("{"):
                    continue
                stats["events"] += 1
                try:
                    ev = json.loads(body)
                    delta = ((ev.get("choices") or [{}])[0].get("delta") or {})
                    if isinstance(delta, dict):
                        if delta.get("reasoning_content"):
                            stats["reasoning"] += 1
                        if delta.get("content"):
                            stats["content"] += 1
                        if delta.get("tool_calls"):
                            stats["tool_calls"] += 1
                    fr = ((ev.get("choices") or [{}])[0].get("finish_reason"))
                    if fr:
                        stats["done"] = True
                except Exception:
                    pass
                yield line
            if getattr(self.valves, "DEBUG_LOG", False):
                log.info(
                    "agent-loop-guard: response events=%d reasoning_deltas=%d content_deltas=%d tool_call_deltas=%d finish=%s",
                    stats["events"],
                    stats["reasoning"],
                    stats["content"],
                    stats["tool_calls"],
                    "yes" if stats["done"] else "no",
                )

    async def _call(self, payload: dict, headers: dict, url: str) -> dict:
        r = await self._client.post(url, json=payload, headers=headers)  # client default 300s
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__=None,
    ):
        messages = body.get("messages", [])
        if not messages:
            return ""

        # --- Reasoning replay (opt-in) -------------------------------------
        # Open WebUI discards the real reasoning text when rebuilding assistant
        # history for OpenAI-compatible models (get_reasoning_format -> None).
        # With REPLAY_REASONING_TEXT on, patch get_reasoning_format so the real
        # text is replayed as reasoning_content on tool-call continuations;
        # otherwise the forcing step below can only inject a single-space
        # placeholder. Installed lazily here (first call installs it before any
        # continuation is rebuilt) and idempotent.
        if getattr(self.valves, "REPLAY_REASONING_TEXT", False):
            _install_reasoning_replay_patch()

        # NOTE: we intentionally do NOT rewrite the "model" field in the
        # upstream response. Open WebUI's frontend persists the assistant
        # message under the Workspace ID the user actually selected
        # (message.model is set client-side at message creation and is never
        # overwritten from the SSE/dict "model"), so rewriting it here did
        # not help Analytics attribution — it only added noise to the stream.
        real_model = body["model"].split(".", 1)[-1]
        headers = {
            "Content-Type": "application/json",
            **self._build_gateway_headers(user=__user__, metadata=__metadata__),
        }
        url = f"{self.valves.GATEWAY_BASE_URL.rstrip('/')}/chat/completions"

        # --- Analyse tool calls ---------------------------------------------
        should_block, bad_tool, kind, total, max_calls = self._analyse(body)

        log.info(
            "Agent Loop Guard → %s (block=%s, kind=%s, tool=%s, total=%s, max=%s)",
            url,
            should_block,
            kind,
            bad_tool,
            total,
            max_calls,
        )

        # --- Block: replace last tool result --------------------------------
        if should_block:
            guard_msg = _build_guard_message(kind, bad_tool, total, max_calls)
            if guard_msg:
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "tool":
                        messages[i]["content"] = guard_msg
                        log.info(
                            "Tool result replaced with guard (kind=%s, tool=%s)",
                            kind,
                            bad_tool,
                        )
                        break

            if __event_emitter__:
                try:
                    if kind == "loop":
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {
                                    "type": "error",
                                    "content": MSG_NOTIFY_LOOP.format(tool=bad_tool),
                                },
                            }
                        )
                    elif kind == "runaway":
                        await __event_emitter__(
                            {
                                "type": "notification",
                                "data": {
                                    "type": "error",
                                    "content": MSG_NOTIFY_RUNAWAY.format(
                                        total=total, max_calls=max_calls
                                    ),
                                },
                            }
                        )
                except Exception:
                    log.warning("Failed to emit event (non-fatal)", exc_info=True)

        # --- Always show counter pill if there are tool calls ---------------
        if __event_emitter__ and max_calls > 0 and total > 0:
            remaining = max(0, max_calls - total)
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": MSG_COUNTER.format(
                                remaining=remaining, max_calls=max_calls
                            ),
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
            except Exception:
                pass

        # --- Apply blocklist -----------------------------------------------
        self._apply_tool_blocklist(body)

        # --- Attached-files cleanup (fail-open, cache-safe) -----------------
        if getattr(self.valves, "ATTACHED_FILES_CLEANUP", True):
            try:
                base_url = (
                    await self._get_public_base_url(__request__)
                    if __request__ is not None
                    else ""
                )
                # Content-hash backstop: two different UUIDs with identical
                # bytes (filter tagged an old copy, core tagged the current
                # upload) collapse here even when the filter's this-turn
                # reuse failed. Fail-open: any resolution error degrades to
                # UUID-only dedup.
                hash_lookup = await _resolve_content_hashes(
                    _collect_image_uuids(messages)
                )
                _cleanup_attached_files(messages, base_url, hash_lookup=hash_lookup)
            except Exception as exc:
                log.warning("attached_files cleanup failed (fail-open): %s", exc)

        # --- DeepSeek reasoning forcing (transport-independent) -------------
        # Forces reasoning_content on every assistant once tool-calling is in
        # scope. Must run here — Open WebUI tool-call continuations bypass
        # filter inlets, so this pipe is the only hop that sees every outbound
        # request to the gateway. Validated against LiteLLM (its own warning,
        # transformation.py) and Bifrost.
        try:
            forced = _force_reasoning_on_gateway_payload(body)
            thinking_stripped = _normalize_thinking_for_gateway(body)
            has_tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
            # All diagnostics are gated behind DEBUG_LOG (default off): with
            # the valve disabled no per-request line is emitted.
            if getattr(self.valves, "DEBUG_LOG", False):
                params = {
                    k: v
                    for k, v in body.items()
                    if k not in ("messages", "tools", "model", "metadata", "files")
                }
                log.info(
                    "agent-loop-guard: forced=%d thinking_stripped=%s "
                    "(model=%s, tools=%s, history_has_tool_calls=%s) "
                    "| messages: %s | params: %s",
                    forced,
                    "yes" if thinking_stripped else "no",
                    real_model,
                    has_tools,
                    _history_has_tool_calls(messages),
                    _messages_summary(messages, verbose=True),
                    params,
                )
                # What does the LAST assistant carry as reasoning_content?
                # R0 (empty) replayed to DeepSeek means the model gets a blank
                # reasoning chain for this turn (LiteLLM warns about exactly
                # this).
                last_rc = None
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        last_rc = m.get("reasoning_content")
                        break
                if isinstance(last_rc, str):
                    preview = last_rc[:80].replace("\n", "\\n")
                    log.info(
                        "agent-loop-guard: last assistant reasoning_content len=%d empty=%s preview=%r",
                        len(last_rc),
                        last_rc == "",
                        preview,
                    )
                elif last_rc is None:
                    log.info("agent-loop-guard: last assistant has NO reasoning_content field")
                else:
                    log.info(
                        "agent-loop-guard: last assistant reasoning_content is non-string (%s)",
                        type(last_rc).__name__,
                    )
        except Exception as exc:
            log.warning("reasoning forcing failed (fail-open): %s", exc)

        # --- Replay-effectiveness watchdog (silent-degradation case) ---------
        # With REPLAY_REASONING_TEXT on, Open WebUI should replay the real
        # reasoning text and `forced` should stay 0 on tool-call continuations.
        # If we still force placeholders on a tool-call history, the patch is
        # installed but ineffective (Open WebUI internals changed in a way the
        # patch does not cover) — the degradation is otherwise silent. Rate-
        # limited so it does not spam on every request.
        if (
            getattr(self.valves, "REPLAY_REASONING_TEXT", False)
            and forced > 0
            and _history_has_tool_calls(messages)
        ):
            _rate_limited_warning(
                "warning",
                "agent-loop-guard: REPLAY_REASONING_TEXT is on but assistant "
                "messages still carry placeholder reasoning (forced=%d on a "
                "tool-call history) — the get_reasoning_format patch appears "
                "ineffective (Open WebUI internals may have changed); enable "
                "DEBUG_LOG to inspect. Degrading to placeholder forcing: %s",
                f"forced={forced}",
            )

        payload = {**body, "model": real_model, "messages": messages}

        if getattr(self.valves, "DEBUG_LOG", False):
            try:
                import json as _json

                log.info(
                    "agent-loop-guard: OUTBOUND url=%s headers=%s payload=%s",
                    url,
                    _json.dumps(headers, default=str),
                    _json.dumps(payload, default=str, ensure_ascii=False)[:4000],
                )
            except Exception as exc:
                log.warning("agent-loop-guard: could not log outbound request: %s", exc)

        try:
            if body.get("stream", False):
                return self._stream(payload, headers, url)
            else:
                return await self._call(payload, headers, url)
        except httpx.HTTPStatusError as e:
            body_preview = ""
            try:
                body_preview = (e.response.text or "")[:2000]
            except Exception:
                pass
            log.error(
                "Gateway returned HTTP %d: %s | body: %s",
                e.response.status_code,
                e,
                body_preview,
            )
            return f"Gateway error: HTTP {e.response.status_code}."
        except httpx.RequestError as e:
            log.error("Gateway unreachable: %s", e)
            return "Gateway unreachable."
        except Exception as e:
            log.error("Unexpected error: %s", e)
            return f"Error: {e}"
