"""Unit tests for the DeepSeek-contract reasoning forcing in agent_loop_guard.

The forcing is the transport-independent part of the (former Bifrost)
reasoning fixes: DeepSeek requires `reasoning_content` on every assistant
message of a tool-calling history — Open WebUI rebuilds assistant messages
without the field on tool-call continuations, and LiteLLM warns
(transformation.py) that a missing field injects a blank reasoning chain.

These tests run against the module without Open WebUI (open_webui imports
are lazy inside the pipe class only).
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_loop_guard import (
    _force_reasoning_content_on_assistant,
    _force_reasoning_on_gateway_payload,
    _history_has_tool_calls,
    _messages_summary,
    Pipe,
)

import httpx  # noqa: E402


def _assistant(**extra) -> dict:
    msg = {"role": "assistant", "content": "thinking...", "reasoning_content": "r"}
    msg.update(extra)
    return msg


def _payload(messages, tools=None) -> dict:
    body = {"model": "deepseek/deepseek-v4-flash", "messages": messages}
    if tools is not None:
        body["tools"] = tools
    return body


# --- _history_has_tool_calls -------------------------------------------------


def test_history_has_tool_calls_detects_tool_calls():
    msgs = [
        {"role": "user", "content": "hi"},
        _assistant(tool_calls=[{"id": "c1", "type": "function", "function": {}}]),
    ]
    assert _history_has_tool_calls(msgs)


def test_history_has_tool_calls_false_without_tool_calls():
    msgs = [
        {"role": "user", "content": "hi"},
        _assistant(),
    ]
    assert not _history_has_tool_calls(msgs)


# --- _force_reasoning_content_on_assistant -----------------------------------


def test_forcing_adds_empty_field_to_bare_assistant():
    msgs = [_assistant()]
    del msgs[0]["reasoning_content"]
    forced = _force_reasoning_content_on_assistant(msgs)
    assert forced == 1
    assert msgs[0]["reasoning_content"] == " "


def test_forcing_preserves_existing_field():
    msgs = [_assistant(reasoning_content="already there")]
    forced = _force_reasoning_content_on_assistant(msgs)
    assert forced == 0
    assert msgs[0]["reasoning_content"] == "already there"


def test_forcing_never_touches_user_system_tool():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        _assistant(),
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    msgs[2].pop("reasoning_content", None)  # bare assistant, like OWUI rebuilds
    original = copy.deepcopy(msgs)
    forced = _force_reasoning_content_on_assistant(msgs)
    assert forced == 1  # only the assistant gets the field
    assert msgs[0] == original[0]
    assert msgs[1] == original[1]
    assert msgs[3] == original[3]


# --- _force_reasoning_on_gateway_payload -------------------------------------


def test_payload_forcing_skipped_without_tool_scope():
    body = _payload([{"role": "user", "content": "hi"}, _assistant()])
    forced = _force_reasoning_on_gateway_payload(body)
    assert forced == 0
    assert "reasoning_content" in body["messages"][1]  # untouched


def test_payload_forcing_with_tools_in_request():
    body = _payload(
        [{"role": "user", "content": "hi"}, _assistant()],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    del body["messages"][1]["reasoning_content"]
    forced = _force_reasoning_on_gateway_payload(body)
    assert forced == 1
    assert body["messages"][1]["reasoning_content"] == " "


def test_payload_forcing_with_tool_call_history():
    body = _payload(
        [
            {"role": "user", "content": "hi"},
            _assistant(tool_calls=[{"id": "c1", "type": "function", "function": {}}]),
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            _assistant(),
        ]
    )
    for m in body["messages"]:
        if m["role"] == "assistant":
            m.pop("reasoning_content", None)
    forced = _force_reasoning_on_gateway_payload(body)
    assert forced == 2
    for m in body["messages"]:
        if m["role"] == "assistant":
            assert m["reasoning_content"] == " "


def test_payload_forcing_deterministic():
    body = _payload(
        [{"role": "user", "content": "hi"}, _assistant()],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )
    first = _force_reasoning_on_gateway_payload(copy.deepcopy(body))
    second = _force_reasoning_on_gateway_payload(copy.deepcopy(body))
    assert first == second


# --- _messages_summary -------------------------------------------------------


# --- shared connection pool -------------------------------------------------


def test_replay_reasoning_text_default_on():
    # The reasoning replay is on by default (A/B-verified richer continuations);
    # disabling it is the opt-out, not the opt-in.
    assert Pipe.Valves().REPLAY_REASONING_TEXT is True


def test_shared_client_pool_configured():
    pipe = Pipe()
    assert isinstance(pipe._client, httpx.AsyncClient)
    # non-stream default budget is 300s, shared across the pool
    assert pipe._client.timeout.connect == 300.0
    assert pipe._client.timeout.read == 300.0
    # pool is per-instance: two pipes do not share one client
    other = Pipe()
    assert other._client is not pipe._client


def test_messages_summary_verbose_flags():
    msgs = [
        {"role": "user", "content": "hi"},
        _assistant(reasoning_content=""),
        _assistant(reasoning_content="abcd"),
        _assistant(tool_calls=[{"id": "c1", "type": "function", "function": {}}]),
    ]
    summary = _messages_summary(msgs, verbose=True)
    assert "user-" in summary
    assert "assistantR0" in summary
    assert "assistantR4" in summary
    assert "assistantTR" in summary
