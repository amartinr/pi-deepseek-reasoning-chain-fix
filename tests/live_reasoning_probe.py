"""Live probe of the Agent Loop Guard reasoning-chaining fix against a real
LiteLLM gateway (http://litellm.private).

Scenario: a tool-call turn. Turn 1 lets the model REASON + CALL get_weather
(authentically). Then the same continuation is replayed under the three
conditions the pipe cares about:

  A) assistant rebuilt WITHOUT reasoning_content   -> Open WebUI rebuild (bug)
  B) assistant with reasoning_content=" "          -> pipe placeholder forcing
  C) assistant with reasoning_content=<REAL text>  -> pipe replay (patch on)

Measures per condition: does the response carry reasoning_content (non-stream
message) and how many reasoning deltas arrive (stream).
"""

import asyncio
import json
import os
import sys

import httpx

BASE = "http://litellm.private"
URL = f"{BASE}/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"


def _load_api_key() -> str:
    """Key from $LITELLM_API_KEY, else from /tmp/.litellm_key (0600)."""
    env = os.environ.get("LITELLM_API_KEY")
    if env:
        return env
    with open("/tmp/.litellm_key") as fh:
        return fh.read().strip()


KEY = _load_api_key()
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns JSON {temp, sky}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. London"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["city"],
            },
        },
    }
]


async def complete(messages, tools=None, stream=False):
    body = {"model": MODEL, "messages": messages, "stream": stream, "temperature": 0.2}
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)) as client:
        r = await client.post(URL, json=body, headers=HEADERS)
        if r.status_code != 200:
            return {"error": r.status_code, "body": r.text[:500]}
        if not stream:
            return r.json()
        # stream: collect deltas
        reasoning_deltas, content_deltas, tool_deltas = 0, 0, 0
        finish = None
        full_reasoning = []
        full_content = []
        async for line in r.aiter_lines():
            line = line.strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                ev = json.loads(line[5:].strip())
            except Exception:
                continue
            choices = ev.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning_deltas += 1
                full_reasoning.append(delta["reasoning_content"])
            if delta.get("content"):
                content_deltas += 1
                full_content.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
        return {
            "reasoning_deltas": reasoning_deltas,
            "content_deltas": content_deltas,
            "tool_call_deltas": tool_deltas,
            "finish": finish,
            "reasoning": "".join(full_reasoning),
            "content": "".join(full_content),
        }


async def main():
    print("=== TURN 1: model reasons + calls get_weather ===", flush=True)
    turn1 = await complete(
        [{"role": "user", "content": "What's the weather like in London today?"}],
        tools=TOOLS,
        stream=True,
    )
    print(json.dumps({k: v for k, v in turn1.items() if k != "reasoning"}, ensure_ascii=False, indent=2))
    print(f"reasoning ({len(turn1.get('reasoning',''))} chars): {turn1.get('reasoning','')[:160]!r}")
    if "error" in turn1:
        print("TURN 1 stream failed — cannot continue", flush=True)
        return

    # The assistant message Open WebUI stores: content + tool_calls.
    # NOTE: with stream=True we do not get the final tool_calls array; do a
    # non-stream turn 1 instead to capture the real tool_call.
    turn1ns = await complete(
        [{"role": "user", "content": "What's the weather like in London today?"}],
        tools=TOOLS,
        stream=False,
    )
    msg = turn1ns["choices"][0]["message"]
    real_reasoning = msg.get("reasoning_content", "")
    tool_calls = msg.get("tool_calls", [])
    print(f"turn1 non-stream: reasoning={len(real_reasoning)} chars, tool_calls={[tc['function']['name'] for tc in tool_calls]}")
    if not tool_calls:
        print("No tool call in turn 1 — aborting", flush=True)
        return
    call_id = tool_calls[0]["id"]
    args = tool_calls[0]["function"].get("arguments") or "{}"
    try:
        args = json.loads(args)
    except Exception:
        pass
    city = args.get("city", "London") if isinstance(args, dict) else "London"
    tool_result = json.dumps({"temp": 28, "sky": "sunny", "city": city})

    assistant_base = {
        "role": "assistant",
        "content": msg.get("content", ""),
        "tool_calls": tool_calls,
    }
    tool_msg = {"role": "tool", "tool_call_id": call_id, "content": tool_result}

    conditions = {
        "A: no reasoning_content (OWUI rebuild)": [{"role": "user", "content": "What's the weather like in London today?"}, {**assistant_base}, tool_msg],
        "B: reasoning_content=' ' (pipe forcing)": [{"role": "user", "content": "What's the weather like in London today?"}, {**assistant_base, "reasoning_content": " "}, tool_msg],
        "C: reasoning_content=<real> (pipe replay)": [{"role": "user", "content": "What's the weather like in London today?"}, {**assistant_base, "reasoning_content": real_reasoning}, tool_msg],
    }

    print("\n=== CONTINUATION under the 3 conditions (stream) ===", flush=True)
    results = {}
    for name, msgs in conditions.items():
        print(f"\n--- {name} ---", flush=True)
        res = await complete(msgs, stream=True)
        results[name] = res
        print(json.dumps({k: v for k, v in res.items() if k not in ("reasoning", "content")}, ensure_ascii=False))
        print(f"  reasoning deltas: {res.get('reasoning_deltas')}  ({len(res.get('reasoning',''))} chars)")
        print(f"  content: {res.get('content','')[:120]!r}")

    print("\n=== SUMMARY ===", flush=True)
    for name, res in results.items():
        if "error" in res:
            print(f"{name}: ERROR {res['error']} {res['body']}")
        else:
            print(
                f"{name}: reasoning_deltas={res.get('reasoning_deltas')} "
                f"reasoning_chars={len(res.get('reasoning',''))} "
                f"finish={res.get('finish')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
