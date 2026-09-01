"""Stub exposing Open WebUI's REAL convert_output_to_messages() (v0.11.1).

The function bodies below are copied verbatim from
backend/open_webui/utils/misc.py @ v0.11.1 (the version the pipe README
validates against), plus the reconcile_tool_pairs() helper it calls.
Only the imports are stubbed: JSONCodec falls back to stdlib json when
orjson is not installed.

Why a stub: the open-webui repo is not available in this environment, and
the test needs the REAL reconstruction logic — a mock would not prove that
the pipe's reasoning_format='reasoning_content' patch actually makes the
assistant history carry the reasoning text.
"""

import json

try:
    import orjson  # type: ignore

    class JSONCodec:
        @staticmethod
        def dumps(data, **kwargs):
            kwargs.pop("ensure_ascii", None)
            return orjson.dumps(data, **kwargs).decode("utf-8")

        @staticmethod
        def loads(data):
            return orjson.loads(data)

except ImportError:
    class JSONCodec:
        @staticmethod
        def dumps(data, **kwargs):
            return json.dumps(data, **kwargs)

        @staticmethod
        def loads(data):
            return json.loads(data)


# ---- verbatim from open_webui/utils/misc.py @ v0.11.1 ----------------------


def get_content_from_message(message: dict) -> str | None:
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text")
    elif content:
        return content

    output_text = get_output_text(message.get("output"))
    return output_text or (content if isinstance(content, str) else None)


def get_output_text(output: list | None) -> str:
    if not isinstance(output, list):
        return ""

    texts = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue

        parts = item.get("content") or []
        if not isinstance(parts, list):
            continue

        text = "".join(
            str(part.get("text"))
            for part in parts
            if isinstance(part, dict) and part.get("text") is not None
        )
        # isspace() avoids the full-string copy strip() would make
        if text and not text.isspace():
            texts.append(text)

    return "\n".join(texts)


def reconcile_tool_pairs(messages: list[dict]) -> list[dict]:
    """Drop unpaired tool_use / tool_result from a reconstructed conversation.

    Stored output can be incomplete — a tool result may be missing (e.g. the
    knowledge base was updated mid-chat, or the call was interrupted), or a
    tool call may be missing while its result survived.  Strict providers
    (Anthropic, AWS Bedrock Converse) reject either direction of mismatch.

    Well-formed output is unaffected: every id pairs, so nothing is stripped.
    """
    completed_tool_call_ids = {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    requested_tool_call_ids = {
        tool_call["id"]
        for message in messages
        for tool_call in message.get("tool_calls") or ()
        if message.get("role") == "assistant" and tool_call.get("id")
    }

    reconciled_messages = []
    for message in messages:
        role = message.get("role")

        # Orphan tool result — no assistant ever claimed this call_id.
        if role == "tool" and message.get("tool_call_id") not in requested_tool_call_ids:
            continue

        # Non-assistant or no tool_calls — pass through unchanged.
        if role != "assistant" or not message.get("tool_calls"):
            reconciled_messages.append(message)
            continue

        # Keep only tool_calls whose id received a tool-role response.
        valid_tool_calls = [
            tool_call
            for tool_call in message["tool_calls"]
            if tool_call.get("id") in completed_tool_call_ids
        ]

        if valid_tool_calls:
            reconciled_messages.append({**message, "tool_calls": valid_tool_calls})
            continue

        # All tool_calls were orphans — keep the message only if it
        # carries meaningful text or reasoning content.
        content = get_content_from_message(message) or ""
        has_meaningful_content = content.strip() if isinstance(content, str) else content
        if has_meaningful_content or message.get("reasoning_content"):
            reconciled_messages.append(
                {key: value for key, value in message.items() if key != "tool_calls"}
            )

    return reconciled_messages


def convert_output_to_messages(
    output: list,
    raw: bool = False,
    reasoning_format: str | None = None,
    flatten_tool_images: bool = False,
) -> list[dict]:
    """
    Convert OR-aligned output items to OpenAI Chat Completion-format messages.

    This reconstructs the full conversation from the stored Responses API-native
    output items, including assistant messages with tool_calls arrays and tool
    role messages.

    Args:
        output: List of OR-aligned output items (Responses API format).
        raw: If True, include code interpreter blocks for LLM re-processing
             follow-ups.
        reasoning_format: How to include reasoning blocks in the output:
            - None: skip reasoning (default, safe for strict providers).
            - ``'thinking'``: set as ``thinking`` top-level field
              (for native Ollama).
            - ``'think_tags'``: wrap in ``<think>`` tags inside content
              (for legacy providers that expect reasoning as tagged content).
            - ``'reasoning_content'``: set as ``reasoning_content`` top-level field
              (for llama.cpp, which routes it via the chat template).
        flatten_tool_images: Move tool output images into a following user
            message for Chat Completions providers.
    """
    if not output or not isinstance(output, list):
        return []

    messages = []
    pending_tool_calls = []
    pending_content = []
    pending_reasoning = []  # Only populated for top-level structured reasoning fields.
    pending_reasoning_details = []
    pending_tool_image_urls = []
    pending_tool_outputs = []
    completed_call_ids = {
        item.get("call_id")
        for item in output
        if item.get("type") == "function_call"
        and item.get("call_id")
        and item.get("status") in {"completed", "failed", "rejected"}
    }
    result_call_ids = {
        item.get("call_id")
        for item in output
        if item.get("type") == "function_call_output" and item.get("call_id")
    }
    function_call_ids = completed_call_ids & result_call_ids

    def flush_pending():
        nonlocal pending_content, pending_tool_calls, pending_reasoning, pending_reasoning_details
        if not pending_content and not pending_tool_calls and not pending_reasoning and not pending_reasoning_details:
            return

        message = {
            "role": "assistant",
            "content": "\n".join(pending_content) if pending_content else "",
            **({"tool_calls": pending_tool_calls} if pending_tool_calls else {}),
        }

        if pending_reasoning:
            if reasoning_format == "thinking":
                message["thinking"] = "\n".join(pending_reasoning)
            else:
                message["reasoning_content"] = "\n".join(pending_reasoning)

        if pending_reasoning_details:
            message["reasoning_details"] = pending_reasoning_details

        messages.append(message)
        pending_content = []
        pending_tool_calls = []
        pending_reasoning = []
        pending_reasoning_details = []

    def flush_tool_images():
        nonlocal pending_tool_image_urls
        if not pending_tool_image_urls:
            return

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are the images from the tool results above. Please analyze them.",
                    },
                    *[
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in pending_tool_image_urls
                    ],
                ],
            }
        )
        pending_tool_image_urls = []

    def flush_tool_outputs():
        nonlocal pending_tool_outputs
        if not pending_tool_outputs:
            return

        flush_pending()
        for output_item in pending_tool_outputs:
            output_parts = output_item.get("output", [])
            content = ""
            image_urls = []
            for part in output_parts:
                if part.get("type") == "input_text":
                    output_text = part.get("text", "")
                    content += (
                        str(output_text) if not isinstance(output_text, str) else output_text
                    )
                elif part.get("type") == "input_image":
                    url = part.get("image_url", "")
                    if url:
                        image_urls.append(url)

            if flatten_tool_images:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": output_item.get("call_id", ""),
                        "content": content,
                    }
                )
                pending_tool_image_urls.extend(image_urls)
            elif image_urls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": output_item.get("call_id", ""),
                        "content": [
                            {"type": "input_text", "text": content},
                            *[{"type": "input_image", "image_url": url} for url in image_urls],
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": output_item.get("call_id", ""),
                        "content": content,
                    }
                )

        pending_tool_outputs = []

    for item in output:
        item_type = item.get("type", "")
        if item_type not in {"function_call", "function_call_output"}:
            flush_tool_outputs()
            flush_tool_images()

        if item_type == "message":
            # Extract text from output_text content parts
            content_parts = item.get("content", [])
            text = ""
            for part in content_parts:
                if part.get("type") == "output_text":
                    text += part.get("text", "")
            if text:
                pending_content.append(text)

        elif item_type == "function_call":
            if item.get("call_id") not in function_call_ids:
                continue

            # Collect tool calls to batch into assistant message
            arguments = item.get("arguments", "{}")
            # Ensure arguments is always a JSON string
            if not isinstance(arguments, str):
                arguments = JSONCodec.dumps(arguments)
            pending_tool_calls.append(
                {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )

        elif item_type == "function_call_output":
            if item.get("call_id") not in function_call_ids:
                continue

            pending_tool_outputs.append(item)

        elif item_type == "reasoning":
            reasoning_details = item.get("reasoning_details") if raw else None
            if reasoning_details:
                reasoning_details = (
                    reasoning_details
                    if isinstance(reasoning_details, list)
                    else [reasoning_details]
                )
                reasoning_details = [
                    detail
                    for detail in reasoning_details
                    if isinstance(detail, dict)
                    and (
                        detail.get("format") != "anthropic-claude-v1"
                        or detail.get("signature")
                    )
                ]
            if not reasoning_format and not reasoning_details:
                continue

            reasoning_text = ""
            source_list = item.get("summary", []) or item.get("content", [])
            for part in source_list:
                if part.get("type") == "output_text":
                    reasoning_text += part.get("text", "")
                elif "text" in part:
                    reasoning_text += part.get("text", "")

            if reasoning_text:
                if reasoning_format == "think_tags":
                    # Legacy tag replay: embed in content with the item's original tags.
                    start_tag = item.get("start_tag", "<think>")
                    end_tag = item.get("end_tag", "</think>")
                    pending_content.append(f"{start_tag}{reasoning_text}{end_tag}")
                elif reasoning_format in {"thinking", "reasoning_content"}:
                    # Native providers: collect for their top-level reasoning field.
                    pending_reasoning.append(reasoning_text)

            if reasoning_details:
                pending_reasoning_details.extend(reasoning_details)

        elif item_type == "open_webui:code_interpreter":
            # Always include code interpreter content so the LLM knows
            # the code was already executed and doesn't retry.
            code = item.get("code", "")
            code_output = item.get("output", "")

            if code:
                pending_content.append(f"<code_interpreter>\n{code}\n</code_interpreter>")

            if code_output:
                if isinstance(code_output, dict):
                    stdout = code_output.get("stdout", "")
                    result = code_output.get("result", "")
                    output_text = stdout or result
                else:
                    output_text = str(code_output)
                if output_text:
                    pending_content.append(
                        f"<code_interpreter_output>\n{output_text}\n</code_interpreter_output>"
                    )

        elif item_type.startswith("open_webui:"):
            # Skip other extension types
            pass

    # Flush remaining content/tool_calls
    flush_tool_outputs()
    flush_tool_images()
    flush_pending()

    return reconcile_tool_pairs(messages)


def _load_owui_misc():
    """Namespace object the tests consume (mirrors the cloned-repo stub).

    SimpleNamespace keeps convert_output_to_messages a PLAIN function (no
    method binding), exactly like the module-level function in the real
    open-webui repo.
    """
    import types

    return types.SimpleNamespace(convert_output_to_messages=convert_output_to_messages)
