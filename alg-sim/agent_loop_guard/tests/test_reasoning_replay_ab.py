"""Unit tests: Open WebUI's assistant-history reconstruction with and without
the reasoning-replay monkey patch (get_reasoning_format -> 'reasoning_content'
for pipe models).

Uses the REAL convert_output_to_messages() from the cloned open-webui repo
(probes/litellm/owui_misc_stub.py), simulating:

- WITHOUT patch: reasoning_format=None (what Open WebUI returns for any
  OpenAI-compatible model: LiteLLM, Bifrost) -> reasoning text is DISCARDED.
- WITH patch:   reasoning_format='reasoning_content' (what the pipe's monkey
  patch makes get_reasoning_format return for pipe models) -> reasoning text
  is replayed in the reasoning_content field.

The fixture output items mirror the Responses-API-aligned storage Open WebUI
builds for a tool-call turn with reasoning (message + reasoning + function
call + tool result).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "probes" / "litellm"))

from owui_misc_stub import _load_owui_misc

_misc = _load_owui_misc()
convert_output_to_messages = _misc.convert_output_to_messages


def _toolcall_output() -> list[dict]:
    """Output items of a turn where the model reasoned, called a tool and
    got the result back (the shape Open WebUI stores and re-reconstructs on
    tool-call continuations)."""
    return [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Let me check the weather in Madrid."}],
        },
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "output_text",
                    "text": "The user asks about Madrid weather. I need today's date first, then call get_weather.",
                }
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_00_abc123",
            "name": "get_date",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_00_abc123",
            "output": [{"type": "input_text", "text": '{"date": "2026-09-01"}'}],
        },
    ]


def _assistant_with_reasoning(messages: list[dict]) -> dict:
    """First reconstructed assistant message (role=assistant)."""
    return next(m for m in messages if m.get("role") == "assistant")


# --- WITHOUT patch (reasoning_format=None) --------------------------------


def test_without_patch_reasoning_text_discarded():
    msgs = convert_output_to_messages(_toolcall_output(), raw=True, reasoning_format=None)
    assistant = _assistant_with_reasoning(msgs)
    assert "reasoning_content" not in assistant
    # content survives, tool_calls survive
    assert assistant["content"] == "Let me check the weather in Madrid."
    assert assistant["tool_calls"][0]["function"]["name"] == "get_date"


# --- WITH patch (reasoning_format='reasoning_content') ---------------------


def test_with_patch_reasoning_text_replayed():
    msgs = convert_output_to_messages(
        _toolcall_output(), raw=True, reasoning_format="reasoning_content"
    )
    assistant = _assistant_with_reasoning(msgs)
    assert assistant["reasoning_content"] == (
        "The user asks about Madrid weather. I need today's date first, then call get_weather."
    )


def test_with_patch_tool_calls_preserved():
    msgs = convert_output_to_messages(
        _toolcall_output(), raw=True, reasoning_format="reasoning_content"
    )
    assistant = _assistant_with_reasoning(msgs)
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["id"] == "call_00_abc123"


# --- multi sub-turn (several tool calls in a row) --------------------------


def _multitool_output() -> list[dict]:
    return [
        {"type": "message", "content": [{"type": "output_text", "text": "Checking date first."}]},
        {
            "type": "reasoning",
            "content": [{"type": "output_text", "text": "Step 1: get today's date."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_date",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": '{"date": "2026-09-01"}'}],
        },
        {"type": "message", "content": [{"type": "output_text", "text": "Now the weather."}]},
        {
            "type": "reasoning",
            "content": [{"type": "output_text", "text": "Step 2: tomorrow is 09-02, call get_weather."}],
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "get_weather",
            "arguments": '{"city": "Madrid", "date": "2026-09-02"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": [{"type": "input_text", "text": '{"temp": 28, "sky": "sunny"}'}],
        },
    ]


def test_without_patch_multitool_no_reasoning_fields():
    msgs = convert_output_to_messages(_multitool_output(), raw=True, reasoning_format=None)
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) == 2  # one assistant per sub-turn
    for a in assistants:
        assert "reasoning_content" not in a


def test_with_patch_multitool_every_subturn_reasons():
    msgs = convert_output_to_messages(
        _multitool_output(), raw=True, reasoning_format="reasoning_content"
    )
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["reasoning_content"] == "Step 1: get today's date."
    assert assistants[1]["reasoning_content"] == "Step 2: tomorrow is 09-02, call get_weather."
    assert assistants[1]["tool_calls"][0]["function"]["name"] == "get_weather"


# --- the monkey patch itself (agent_loop_guard._install_reasoning_replay_patch)

import types  # noqa: E402

import agent_loop_guard as alg  # noqa: E402


def _fresh_middleware():
    """Inject a fake open_webui.utils.middleware whose get_reasoning_format
    returns None (what Open WebUI does for OpenAI-compatible models)."""
    mw = types.ModuleType("open_webui.utils.middleware")

    def fake_get_reasoning_format(model):
        return None

    mw.get_reasoning_format = fake_get_reasoning_format
    sys.modules["open_webui.utils.middleware"] = mw
    return mw


def test_patch_returns_reasoning_content_for_pipe_models():
    alg._REASONING_REPLAY_PATCHED = False
    mw = _fresh_middleware()
    assert alg._install_reasoning_replay_patch() is True
    # pipe model (owned_by=openai + pipe key): replayed as reasoning_content
    assert (
        mw.get_reasoning_format({"owned_by": "openai", "pipe": True})
        == "reasoning_content"
    )
    # plain openai model (no pipe key): original behavior preserved
    assert mw.get_reasoning_format({"owned_by": "openai"}) is None
    # ollama / llama.cpp: original behavior preserved
    assert mw.get_reasoning_format({"owned_by": "ollama"}) is None


def test_patch_idempotent_and_single_wrapper():
    alg._REASONING_REPLAY_PATCHED = False
    mw = _fresh_middleware()
    assert alg._install_reasoning_replay_patch() is True
    assert alg._install_reasoning_replay_patch() is True  # no-op via marker
    assert getattr(mw.get_reasoning_format, "_alg_reasoning_patched", False)


def test_patch_fails_open_when_middleware_missing():
    alg._REASONING_REPLAY_PATCHED = False
    sys.modules.pop("open_webui.utils.middleware", None)
    # patch must not raise and must report inactive (forcing fallback)
    assert alg._install_reasoning_replay_patch() is False
    alg._REASONING_REPLAY_PATCHED = False  # restore for later tests
