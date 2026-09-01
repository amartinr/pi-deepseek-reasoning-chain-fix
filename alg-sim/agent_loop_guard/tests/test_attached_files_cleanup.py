"""Unit tests for the cache-safe <attached_files> cleanup in agent_loop_guard.

The cleanup is a pure function of the payload: it must be deterministic,
idempotent, and keep the history prefix byte-stable between consecutive
turns (that is what lets LLM prefix caches hit). These tests run against
the module without Open WebUI (open_webui imports are lazy inside the
pipe class only).
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_loop_guard import (
    _build_block,
    _cleanup_attached_files,
    _collect_image_uuids,
    _dedupe_tags,
    _file_dedup_key,
    _normalize_tag,
    _parse_file_tags,
)

BASE = "http://open-webui.private"
ABS = f"{BASE}/api/v1/files"


def _core_block(file_id: str, name: str = "") -> str:
    """The core's add_file_context() format: relative URL, own attrs."""
    extra = f' content_type="image/png" name="{name}"' if name else ""
    return (
        "<attached_files>\n"
        f'<file type="image" id="{file_id}" url="/api/v1/files/{file_id}/content"{extra}/>\n'
        "</attached_files>\n\n"
    )


def _union_block(*file_ids: str) -> str:
    """The image_filter's union block: absolute URLs, one tag per file."""
    lines = "\n".join(
        f'<file type="image" id="{fid}" url="{ABS}/{fid}/content"/>' for fid in file_ids
    )
    return f"<attached_files>\n{lines}\n</attached_files>\n\n"


def _user(parts: list[str]) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": p} for p in parts]}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_single_core_block():
    tags = _parse_file_tags(_core_block("f1", "one.png") + "Hi")
    assert len(tags) == 1
    assert dict(tags[0]["attrs"]) == {
        "type": "image",
        "id": "f1",
        "url": "/api/v1/files/f1/content",
        "content_type": "image/png",
        "name": "one.png",
    }


def test_parse_two_blocks_in_one_text():
    text = _core_block("f1") + _union_block("f1", "f2") + "Hi"
    tags = _parse_file_tags(text)
    assert [dict(t["attrs"])["id"] for t in tags] == ["f1", "f1", "f2"]


def test_parse_ignores_block_without_file_tags():
    assert _parse_file_tags("<attached_files>\n</attached_files>\n\nHi") == []


def test_parse_mixed_part_block_plus_text():
    tags = _parse_file_tags(_union_block("f1", "f2") + "User text here")
    assert len(tags) == 2


# ---------------------------------------------------------------------------
# Dedup keys
# ---------------------------------------------------------------------------


def _tag(attrs):
    return {"raw": "<file/>", "attrs": list(attrs.items())}


def _img_tag(attrs):
    # Image tags carry type/content_type so they participate in cleanup.
    return _tag({"type": "image", **attrs})


def test_dedup_key_absolute_and_relative_same_path():
    assert _file_dedup_key(_tag({"url": "/api/v1/files/abc/content"})) == _file_dedup_key(
        _tag({"url": "http://host/api/v1/files/abc/content"})
    )


def test_dedup_key_id_only():
    assert _file_dedup_key(_img_tag({"id": "abc"})) == "id:abc"
    assert _file_dedup_key(_img_tag({"id": "abc"})) == _file_dedup_key(
        _img_tag({"url": "/api/v1/files/abc/content"})
    )


def test_dedup_key_bare_uuid_url():
    assert _file_dedup_key(_img_tag({"url": "d264f103-0fb6-480e-a310-dcb62f10ef30"})) == (
        "id:d264f103-0fb6-480e-a310-dcb62f10ef30"
    )


def test_dedup_key_external_url():
    assert _file_dedup_key(_img_tag({"url": "https://cdn.example.com/img.png"})) == (
        "url:https://cdn.example.com/img.png"
    )


def test_dedup_key_placeholder_never_deduped():
    assert _file_dedup_key(_tag({"url": "(base64 stripped)"})) == ""
    assert _file_dedup_key(_tag({"id": "(base64 stripped)"})) == ""


def test_dedupe_tags_keeps_placeholders():
    tags = [
        _img_tag({"url": "/api/v1/files/abc/content"}),
        _img_tag({"url": "http://h/api/v1/files/abc/content"}),
        _img_tag({"url": "(base64 stripped)"}),
    ]
    kept = _dedupe_tags(tags, set())
    assert [dict(t["attrs"])["url"] for t in kept] == ["/api/v1/files/abc/content", "(base64 stripped)"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_relative_url():
    tag = _tag({"type": "image", "id": "f1", "url": "/api/v1/files/f1/content"})
    assert _normalize_tag(tag, BASE) == (
        '<file type="image" id="f1" url="http://open-webui.private/api/v1/files/f1/content"/>'
    )


def test_normalize_canonicalizes_file_urls_keeps_data_uris():
    # Absolute /api/v1/files/{id}/content image URLs are canonicalized to
    # our format (id + our absolute URL); data: URIs (no UUID) pass
    # through as-is.
    out = _normalize_tag(_img_tag({"url": "http://h/api/v1/files/a/content"}), BASE)
    assert 'id="a"' in out
    assert out.endswith('url="http://open-webui.private/api/v1/files/a/content"/>')
    assert _normalize_tag(_img_tag({"url": "data:image/png;base64,AAA"}), BASE).endswith(
        'url="data:image/png;base64,AAA"/>'
    )


def test_normalize_without_base_url_emits_id_format():
    # No base URL available → still emit our format (id + relative URL),
    # so view_file keeps working.
    tag = _tag({"type": "image", "id": "f1", "url": "/api/v1/files/f1/content"})
    assert _normalize_tag(tag, "") == '<file type="image" id="f1" url="/api/v1/files/f1/content"/>'


def test_build_block_core_format():
    block = _build_block([_tag({"type": "image", "id": "f1", "url": "/api/v1/files/f1/content"})], BASE)
    assert block == (
        "<attached_files>\n"
        '<file type="image" id="f1" url="http://open-webui.private/api/v1/files/f1/content"/>\n'
        "</attached_files>\n\n"
    )
    assert _build_block([], BASE) == ""


def test_raw_core_form_normalized_to_our_format():
    # This deployment's core emits <file type="file" url="{uuid}" .../>
    # (bare UUID, no /api/v1/files/ path). The pipe must re-emit it in
    # OUR format: id + absolute URL, so view_file (id) and ComfyUI (URL)
    # both work.
    raw = (
        "<attached_files>\n"
        '<file type="file" url="79cb1456-2b61-4b84-9769-787f5f6eb859" content_type="image/png" name="home.png"/>\n'
        "</attached_files>\n\n"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": raw}, {"type": "text", "text": "What do you see?"}]}
    ]
    _cleanup_attached_files(messages, BASE)
    assert messages[0]["content"][0]["text"] == (
        "<attached_files>\n"
        '<file type="image" id="79cb1456-2b61-4b84-9769-787f5f6eb859" '
        'url="http://open-webui.private/api/v1/files/79cb1456-2b61-4b84-9769-787f5f6eb859/content" '
        'content_type="image/png" name="home.png"/>\n'
        "</attached_files>\n\n"
    )


def test_non_image_file_untouched():
    # A PDF upload: the image_filter passes it through (images only), the
    # core adds its <file> tag, and the pipe's cleanup must NOT touch it —
    # only image tags are deduplicated/re-written. The PDF keeps its raw
    # form exactly as the core emitted it.
    raw = (
        "<attached_files>\n"
        '<file type="file" url="11111111-2222-4333-8444-555555555555" content_type="application/pdf" name="doc.pdf"/>\n'
        "</attached_files>\n\n"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": raw}, {"type": "text", "text": "Read this?"}]}
    ]
    original = copy.deepcopy(messages)
    _cleanup_attached_files(messages, BASE)
    assert messages == original  # byte-identical: non-image tags untouched


def test_image_and_non_image_coexist():
    # An image and a PDF in the same message: the image is normalized to
    # our format; the PDF is not deduplicated nor rewritten to our format —
    # it keeps its type="file", its id, and its content_type/name. Its
    # relative URL becomes absolute (in the re-emitted block), which is
    # harmless and useful.
    text = (
        "<attached_files>\n"
        '<file type="image" id="img1" url="/api/v1/files/img1/content"/>\n'
        '<file type="file" id="pdf1" url="/api/v1/files/pdf1/content" content_type="application/pdf" name="doc.pdf"/>\n'
        "</attached_files>\n\nHi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE)
    content = messages[0]["content"]
    # image → our canonical format (id + absolute URL)
    assert 'type="image" id="img1" url="http://open-webui.private/api/v1/files/img1/content"' in content
    # pdf → not deduplicated, keeps its identity and metadata
    assert 'type="file" id="pdf1"' in content
    assert 'content_type="application/pdf"' in content and 'name="doc.pdf"' in content
    assert content.count("<file") == 2  # both still present
    assert content.count('id="img1"') == 1 and content.count('id="pdf1"') == 1


def test_raw_and_filter_form_dedup_to_one_tag_keeping_ours():
    # Both forms of the same UUID in the same message → ONE tag, our format.
    raw = '<file type="file" url="79cb1456-2b61-4b84-9769-787f5f6eb859" content_type="image/png" name="home.png"/>'
    ours = (
        '<file type="image" id="79cb1456-2b61-4b84-9769-787f5f6eb859" '
        'url="http://open-webui.private/api/v1/files/79cb1456-2b61-4b84-9769-787f5f6eb859/content"/>'
    )
    text = (
        "<attached_files>\n" + raw + "\n</attached_files>\n\n"
        "<attached_files>\n" + ours + "\n</attached_files>\n\n" + "Hi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE)
    content = messages[0]["content"]
    assert content.count("<file") == 1
    assert 'id="79cb1456-2b61-4b84-9769-787f5f6eb859"' in content
    assert content.count("url=") == 1
    assert content.count("(base64 stripped)") == 0


def test_raw_core_form_normalized_to_our_format():
    # This deployment's core emits <file type="file" url="{uuid}" .../>
    # (bare UUID, no /api/v1/files/ path). The pipe must re-emit it in
    # OUR format: id + absolute URL, so view_file (id) and ComfyUI (URL)
    # both work.
    raw = (
        "<attached_files>\n"
        '<file type="file" url="79cb1456-2b61-4b84-9769-787f5f6eb859" content_type="image/png" name="home.png"/>\n'
        "</attached_files>\n\n"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": raw}, {"type": "text", "text": "What do you see?"}]}
    ]
    _cleanup_attached_files(messages, BASE)
    assert messages[0]["content"][0]["text"] == (
        "<attached_files>\n"
        '<file type="image" id="79cb1456-2b61-4b84-9769-787f5f6eb859" '
        'url="http://open-webui.private/api/v1/files/79cb1456-2b61-4b84-9769-787f5f6eb859/content" '
        'content_type="image/png" name="home.png"/>\n'
        "</attached_files>\n\n"
    )


# ---------------------------------------------------------------------------
# Cleanup semantics
# ---------------------------------------------------------------------------


def test_historical_core_block_normalized_when_alone():
    messages = [_user([_core_block("f1"), "First?"])]
    _cleanup_attached_files(messages, BASE)
    assert messages[0]["content"][0]["text"] == _core_block("f1").replace(
        "/api/v1/files/f1/content", f"{ABS}/f1/content"
    )
    assert messages[0]["content"][1]["text"] == "First?"


def test_last_message_collapses_duplicate_blocks_from_filter_and_core():
    # The current turn's message carries the same upload twice: once from
    # the core's add_file_context() and once from the image_filter's
    # current-turn block (same UUID — the filter converges on the current
    # upload since v2.12.2). Dedup is per-message: the two blocks collapse
    # into one tag, while turn-1's file keeps its own block in u1.
    messages = [
        _user([_core_block("f1"), "First?"]),
        {"role": "assistant", "content": "Answer 1"},
        _user([_core_block("f2"), _union_block("f2") + "Second?"]),
    ]
    _cleanup_attached_files(messages, BASE)

    u2_text = "\n".join(p["text"] for p in messages[2]["content"] if p.get("type") == "text")
    assert "id=\"f2\"" in u2_text
    assert u2_text.count("<attached_files>") == 1  # blocks collapsed into one
    assert u2_text.count("id=\"f2\"") == 1         # duplicate tag dropped within the message
    assert u2_text.endswith("Second?")

    # u1 keeps its own file — nothing cross-message is deduplicated.
    u1_text = "\n".join(p["text"] for p in messages[0]["content"] if p.get("type") == "text")
    assert "id=\"f1\"" in u1_text


def test_re_attached_file_visible_in_later_turn():
    # Deliberate re-upload of the same file in a later turn (str content):
    # the pipe must NOT deduplicate it away — the current turn gets its
    # own block with the re-uploaded file, so the agent is aware it was
    # uploaded again.
    messages = [
        {"role": "user", "content": _core_block("f1") + "First?"},
        {"role": "user", "content": _core_block("f1") + _union_block("f1") + "Same file again."},
    ]
    _cleanup_attached_files(messages, BASE)
    u2 = messages[1]["content"]
    assert "id=\"f1\"" in u2  # visible again in its turn
    assert u2.count("<attached_files>") == 1  # filter+core blocks collapsed into one
    assert u2.endswith("Same file again.")


def test_re_attached_file_tagged_in_each_turn():
    # Same file re-attached in a later turn: tagged in EACH turn (its own
    # block per message). No cross-message dedup.
    messages = [
        {"role": "user", "content": _core_block("f1") + "First?"},
        {"role": "user", "content": _core_block("f1") + "Re-attach?"},
    ]
    _cleanup_attached_files(messages, BASE)
    assert "id=\"f1\"" in messages[0]["content"]
    assert "id=\"f1\"" in messages[1]["content"]  # re-attach stays visible


def test_placeholder_preserved_and_not_deduped():
    placeholder_block = "<attached_files>\n<file type=\"image\" url=\"(base64 stripped)\"/>\n</attached_files>\n\n"
    messages = [
        _user([_core_block("f1"), "a"]),
        _user([_union_block("f1", "f2"), placeholder_block + "b"]),
    ]
    _cleanup_attached_files(messages, BASE)
    u2 = "\n".join(p["text"] for p in messages[1]["content"] if p.get("type") == "text")
    assert u2.count("(base64 stripped)") == 1
    # Re-presented files stay in u2 — there is no cross-message dedup.
    assert "id=\"f1\"" in u2
    assert "id=\"f2\"" in u2


def test_empty_content_after_strip_gets_empty_text_part():
    messages = [{"role": "user", "content": [{"type": "text", "text": _core_block("f1")}]}]
    _cleanup_attached_files(messages, BASE)
    assert messages[0]["content"][0]["text"].startswith("<attached_files>")  # f1 kept (earliest)


def test_non_user_messages_untouched():
    messages = [
        {"role": "system", "content": "sys"},
        _user([_core_block("f1"), "a"]),
        {"role": "assistant", "content": "ans"},
        {"role": "tool", "content": _core_block("f1")},  # never injected into tool msgs, but must not be touched
    ]
    _cleanup_attached_files(messages, BASE)
    assert messages[3]["content"] == _core_block("f1")


def test_idempotent():
    messages = [
        _user([_core_block("f1"), "First?"]),
        _user([_core_block("f2"), _union_block("f1", "f2") + "Second?"]),
    ]
    _cleanup_attached_files(messages, BASE)
    once = copy.deepcopy(messages)
    _cleanup_attached_files(messages, BASE)
    assert messages == once


def test_no_blocks_noop():
    messages = [{"role": "user", "content": [{"type": "text", "text": "plain"}]}]
    original = copy.deepcopy(messages)
    _cleanup_attached_files(messages, BASE)
    assert messages == original


def test_fail_open_on_malformed_input():
    messages = [
        {"role": "user", "content": None},
        {"role": "user", "content": 42},
        {"role": "user"},
        "not-a-dict",
    ]
    _cleanup_attached_files(messages, BASE)  # must not raise


# ---------------------------------------------------------------------------
# Cache-safety regression: the history prefix must be byte-stable
# ---------------------------------------------------------------------------
#
# Real conversation, one new file per turn. The image_filter's union block
# moves to the last user message every turn; the core's per-message blocks
# stay put. Cleanup must leave every shared history message byte-identical
# between turn N and turn N+1.


def _turn_payload(include_next_turn: bool) -> list[dict]:
    """Payload as seen in turn 3 (history up to u3) and in turn 4 (same
    history + a3 + u4). Each turn's message carries its own file in a core
    block; the LAST message additionally carries the image_filter's
    current-turn block for the SAME file (duplicate tag, absolute URL) —
    the filter never touches historical messages (since v2.12.0)."""
    messages = [
        _user([_core_block("f1"), "First?"]),
        {"role": "assistant", "content": "Answer 1"},
        _user([_core_block("f2"), "Second?"]),
        {"role": "assistant", "content": "Answer 2"},
    ]
    if include_next_turn:
        # turn 4: history u1..u3 unchanged (core blocks only); u4 is the
        # current turn with its own file from both sources
        messages.append(_user([_core_block("f3"), "Third?"]))
        messages.append({"role": "assistant", "content": "Answer 3"})
        messages.append(_user([_core_block("f4"), _union_block("f4") + "Fourth?"]))
    else:
        # turn 3: u3 is the current turn (core + filter for its own file)
        messages.append(_user([_core_block("f3"), _union_block("f3") + "Third?"]))
    return messages


def test_history_prefix_byte_stable_between_turns():
    """The core cache-safety guarantee.

    Turn 3 and turn 4 share the history u1, a1, u2, a2, u3. After cleanup
    every shared message must be byte-identical — otherwise the provider's
    prefix cache misses on the whole conversation. u3 is the interesting
    one: in turn 3 it carries the filter's current-turn block (core f3 +
    filter f3), in turn 4 only the core block — per-message dedup makes
    both collapse to the same single tag, so u3 is byte-identical too.
    """
    turn3 = _turn_payload(include_next_turn=False)
    turn4 = _turn_payload(include_next_turn=True)

    _cleanup_attached_files(turn3, BASE)
    _cleanup_attached_files(turn4, BASE)

    assert turn3[0] == turn4[0]  # u1
    assert turn3[1] == turn4[1]  # a1
    assert turn3[2] == turn4[2]  # u2
    assert turn3[3] == turn4[3]  # a2
    assert turn3[4] == turn4[4]  # u3 — core+filter in turn 3 vs core only in turn 4 → same block

    # Sanity: each historical message keeps exactly its own file.
    u2 = "\n".join(p["text"] for p in turn3[2]["content"] if p.get("type") == "text")
    u3 = "\n".join(p["text"] for p in turn3[4]["content"] if p.get("type") == "text")
    assert "id=\"f2\"" in u2 and "id=\"f1\"" not in u2
    assert "id=\"f3\"" in u3 and "id=\"f1\"" not in u3 and "id=\"f2\"" not in u3
    # The last message of turn 4 keeps its own (new) file, collapsed to one tag.
    u4 = "\n".join(p["text"] for p in turn4[6]["content"] if p.get("type") == "text")
    assert "id=\"f4\"" in u4 and "id=\"f1\"" not in u4 and "id=\"f2\"" not in u4 and "id=\"f3\"" not in u4
    assert u4.count("id=\"f4\"") == 1


def test_history_prefix_stable_without_base_url():
    turn3 = _turn_payload(include_next_turn=False)
    turn4 = _turn_payload(include_next_turn=True)
    _cleanup_attached_files(turn3, "")
    _cleanup_attached_files(turn4, "")
    assert turn3[2] == turn4[2]
    assert turn3[4] == turn4[4]


# ---------------------------------------------------------------------------
# Content-hash backstop (v2.3.0)
# ---------------------------------------------------------------------------
#
# 2026-08-01 incident: on a single `+` upload turn the model reported "two
# images". The core's add_file_context() tagged the CURRENT upload
# (76680237...) and the image_filter tagged an OLDER identical copy
# (79cb1456...) — two different UUIDs, so UUID dedup kept both. The pipe is
# the last code before the provider, so it also dedups by content
# (`hash_lookup`: uuid -> meta["file_hash"]) and collapses the pair.


def test_content_hash_dedup_collapses_two_uuids_same_bytes():
    upload = "76680237-1167-4692-894d-2de4e02a5b5b"
    old_copy = "79cb1456-2b61-4b84-9769-787f5f6eb859"
    digest = "a" * 64
    text = (
        "<attached_files>\n"
        f'<file type="image" id="{upload}" url="/api/v1/files/{upload}/content" name="home.png"/>\n'
        "</attached_files>\n\n"
        "<attached_files>\n"
        f'<file type="image" id="{old_copy}" url="{ABS}/{old_copy}/content"/>\n'
        "</attached_files>\n\n"
        "Hi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE, hash_lookup={upload: digest, old_copy: digest})
    content = messages[0]["content"]
    assert content.count("<file") == 1
    assert f'id="{upload}"' in content  # first occurrence wins (the upload)
    assert f'id="{old_copy}"' not in content


def test_content_hash_dedup_keeps_different_content():
    a, b = "76680237-1167-4692-894d-2de4e02a5b5b", "79cb1456-2b61-4b84-9769-787f5f6eb859"
    text = (
        "<attached_files>\n"
        f'<file type="image" id="{a}" url="{ABS}/{a}/content"/>\n'
        f'<file type="image" id="{b}" url="{ABS}/{b}/content"/>\n'
        "</attached_files>\n\nHi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE, hash_lookup={a: "a" * 64, b: "b" * 64})
    assert messages[0]["content"].count("<file") == 2


def test_content_hash_dedup_scoped_per_message():
    # Content-hash dedup applies WITHIN a message only (the filter and the
    # core tagging the same upload in one turn). The same content uploaded
    # again in a LATER turn (deliberate re-upload) is kept — it is not
    # deduplicated across messages.
    messages = [
        _user([_core_block("f1"), "First?"]),
        _user([_core_block("f2"), "Second?"]),  # same bytes as f1, different UUID
    ]
    _cleanup_attached_files(messages, BASE, hash_lookup={"f1": "a" * 64, "f2": "a" * 64})
    assert "id=\"f1\"" in messages[0]["content"][0]["text"]
    u2 = "\n".join(p["text"] for p in messages[1]["content"] if p.get("type") == "text")
    assert "id=\"f2\"" in u2  # re-upload kept — not deduped across turns


def test_content_hash_dedup_skips_non_image():
    text = (
        "<attached_files>\n"
        '<file type="file" id="pdf1" url="/api/v1/files/pdf1/content" name="a.pdf" content_type="application/pdf"/>\n'
        '<file type="file" id="pdf2" url="/api/v1/files/pdf2/content" name="b.pdf" content_type="application/pdf"/>\n'
        "</attached_files>\n\nHi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE, hash_lookup={"pdf1": "a" * 64, "pdf2": "a" * 64})
    assert messages[0]["content"].count("<file") == 2  # non-image tags never collapsed


def test_content_hash_dedup_miss_falls_back_to_uuid_only():
    text = _union_block("f1", "f2") + "Hi"
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE, hash_lookup={})  # resolution failed → UUID-only
    assert messages[0]["content"].count("<file") == 2


def test_content_hash_dedup_idempotent():
    upload = "76680237-1167-4692-894d-2de4e02a5b5b"
    old_copy = "79cb1456-2b61-4b84-9769-787f5f6eb859"
    digest = "a" * 64
    text = (
        "<attached_files>\n"
        f'<file type="image" id="{upload}" url="{ABS}/{upload}/content"/>\n'
        f'<file type="image" id="{old_copy}" url="{ABS}/{old_copy}/content"/>\n'
        "</attached_files>\n\nHi"
    )
    messages = [{"role": "user", "content": text}]
    _cleanup_attached_files(messages, BASE, hash_lookup={upload: digest, old_copy: digest})
    once = copy.deepcopy(messages)
    _cleanup_attached_files(messages, BASE, hash_lookup={upload: digest, old_copy: digest})
    assert messages == once


def test_re_upload_same_content_not_deduped_across_turns():
    """Re-uploading the same image (identical content, NEW UUID from the
    `+` button) in a later turn: the pipe must NOT drop the new tag.

    Regression for the reported bug: the re-upload was invisible to the
    agent (cross-message content-hash dedup) while the `+` upload still
    persisted a duplicate on disk — the worst of both worlds. The disk
    copy is core behaviour (upload happens before the pipeline); the
    invisibility is fixed here: the current turn keeps its own block with
    the re-uploaded file, and the original keeps its block in u1.
    """
    f1 = "76680237-1167-4692-894d-2de4e02a5b5b"
    f2 = "79cb1456-2b61-4b84-9769-787f5f6eb859"
    digest = "a" * 64
    messages = [
        {"role": "user", "content": _core_block(f1) + "First?"},
        {"role": "user", "content": _core_block(f2) + _union_block(f2) + "Same image again."},
    ]
    _cleanup_attached_files(messages, BASE, hash_lookup={f1: digest, f2: digest})
    assert f'id="{f1}"' in messages[0]["content"]  # original keeps its block
    u2 = messages[1]["content"]
    assert f'id="{f2}"' in u2          # the re-upload stays visible
    assert u2.count("<file") == 1      # filter+core collapsed within u2 only
    assert f'id="{f1}"' not in u2


def test_re_upload_same_content_cache_safe_prefix():
    """The re-upload must not destabilize the shared history: after the
    re-upload turn, a later plain turn renders the same bytes for every
    shared message (prefix cache keeps hitting)."""
    digest = "a" * 64
    turn2 = [
        {"role": "user", "content": _core_block("f1") + "First?"},
        {"role": "assistant", "content": "Answer 1"},
        # current turn: re-upload of f1's content under a new UUID
        {"role": "user", "content": _core_block("f2") + _union_block("f2") + "Same image again."},
    ]
    turn3 = [
        {"role": "user", "content": _core_block("f1") + "First?"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": _core_block("f2") + "Same image again."},
        {"role": "assistant", "content": "Answer 2"},
        {"role": "user", "content": [{"type": "text", "text": "Plain follow-up."}]},
    ]
    _cleanup_attached_files(turn2, BASE, hash_lookup={"f1": digest, "f2": digest})
    _cleanup_attached_files(turn3, BASE, hash_lookup={"f1": digest, "f2": digest})
    assert turn2[0] == turn3[0]  # u1
    assert turn2[1] == turn3[1]  # a1
    assert turn2[2] == turn3[2]  # u2 — re-upload turn vs its later stored form
    assert "id=\"f2\"" in turn3[2]["content"]  # the re-upload is still visible later


def test_collect_image_uuids():
    text = (
        "<attached_files>\n"
        f'<file type="image" id="f1" url="{ABS}/f1/content"/>\n'
        '<file type="file" id="pdf1" url="/api/v1/files/pdf1/content" name="doc.pdf" content_type="application/pdf"/>\n'
        "</attached_files>\n\n"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": text}]},
        {"role": "user", "content": _core_block("f2") + _core_block("f1")},  # duplicate f1
        {"role": "assistant", "content": "x"},
    ]
    assert _collect_image_uuids(messages) == ["f1", "f2"]
