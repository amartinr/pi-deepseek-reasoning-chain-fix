"""Repeatable live A/B of the reasoning-chaining fix against http://litellm.private.

Extends live_reasoning_probe.py with:
  - N repetitions per condition (variance control)
  - reasoning TEXT captured per condition (continuity vs re-derivation)
  - a 2-step tool chain (get_date -> get_weather) to exercise chaining across
    multiple continuations
"""

import asyncio
import json
import os
import statistics

import httpx

BASE = "http://litellm.private"
URL = f"{BASE}/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"


def _load_api_key() -> str:
    env = os.environ.get("LITELLM_API_KEY")
    if env:
        return env
    with open("/tmp/.litellm_key") as fh:
        return fh.read().strip()


HEADERS = {"Authorization": f"Bearer {_load_api_key()}", "Content-Type": "application/json"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns JSON {temp, sky}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get today's date. Returns JSON {date}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def complete(messages, tools=None, stream=True):
    body = {"model": MODEL, "messages": messages, "stream": stream, "temperature": 0.2}
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)) as client:
        r = await client.post(URL, json=body, headers=HEADERS)
        if r.status_code != 200:
            return {"error": r.status_code, "body": r.text[:400]}
        if not stream:
            return r.json()
        reasoning, content = [], []
        finish = None
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
                reasoning.append(delta["reasoning_content"])
            if delta.get("content"):
                content.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
        return {
            "finish": finish,
            "reasoning_deltas": len(reasoning),
            "reasoning": "".join(reasoning),
            "content": "".join(content),
        }


async def run_turn1() -> dict:
    """Authentic turn 1: model reasons + calls get_weather (non-stream for the tool_calls array)."""
    r = await complete(
        [{"role": "user", "content": "What's the weather like in London today?"}],
        tools=TOOLS,
        stream=False,
    )
    msg = r["choices"][0]["message"]
    return {
        "reasoning": msg.get("reasoning_content", ""),
        "tool_calls": msg.get("tool_calls", []),
        "content": msg.get("content", ""),
    }


def continuation_histories(turn1: dict) -> dict[str, list]:
    """Build the 3 continuation conditions from an authentic turn 1.
    Emits one tool message per tool_call (Open WebUI does the same)."""
    tool_calls = turn1["tool_calls"]
    tool_msgs = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        if name == "get_date":
            result = json.dumps({"date": "2026-09-02"})
        else:
            result = json.dumps({"temp": 28, "sky": "sunny", "city": args.get("city", "London")})
        tool_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    assistant = {"role": "assistant", "content": turn1["content"], "tool_calls": tool_calls}
    user = {"role": "user", "content": "What's the weather like in London today?"}
    return {
        "A:no_reasoning_content": [user, {**assistant}, *tool_msgs],
        "B:placeholder=' '": [user, {**assistant, "reasoning_content": " "}, *tool_msgs],
        "C:real_replay": [user, {**assistant, "reasoning_content": turn1["reasoning"]}, *tool_msgs],
    }


async def run_condition(name: str, msgs: list, reps: int) -> dict:
    stats = {"reasoning_chars": [], "reasoning_deltas": [], "content_chars": []}
    samples = []
    for i in range(reps):
        res = await complete(msgs, tools=TOOLS, stream=True)  # tools travel on continuations (OWUI does)
        if "error" in res:
            stats["error"] = res
            break
        stats["reasoning_chars"].append(len(res["reasoning"]))
        stats["reasoning_deltas"].append(res["reasoning_deltas"])
        stats["content_chars"].append(len(res["content"]))
        samples.append(res["reasoning"])
    out = {
        "reasoning_chars_median": statistics.median(stats["reasoning_chars"]) if stats["reasoning_chars"] else None,
        "reasoning_deltas_median": statistics.median(stats["reasoning_deltas"]) if stats["reasoning_deltas"] else None,
        "content_chars_median": statistics.median(stats["content_chars"]) if stats["content_chars"] else None,
    }
    if "error" in stats:
        out["error"] = stats["error"]
    # first sample's reasoning text, truncated
    out["reasoning_sample"] = samples[0][:220] if samples else ""
    return out


async def two_step_chain() -> dict:
    """Full 2-step tool chain. Step1: get_date. Step2 continuation (rebuilt
    assistant WITHOUT reasoning_content vs WITH placeholder vs WITH real)."""
    step1 = await complete(
        [{"role": "user", "content": "What's the weather like in London tomorrow?"}],
        tools=TOOLS,
        stream=False,
    )
    m1 = step1["choices"][0]["message"]
    if not m1.get("tool_calls"):
        return {"error": "step1 produced no tool call", "raw": {k: v for k, v in m1.items() if k != "reasoning_content"}}
    tc1 = m1["tool_calls"][0]
    r1 = m1.get("reasoning_content", "")
    tool1 = {"role": "tool", "tool_call_id": tc1["id"], "content": json.dumps({"date": "2026-09-02"})}
    asst1 = {"role": "assistant", "content": m1.get("content", ""), "tool_calls": m1["tool_calls"]}
    user = {"role": "user", "content": "What's the weather like in London tomorrow?"}

    conds = {
        "A:no_reasoning_content": [user, {**asst1}, tool1],
        "B:placeholder=' '": [user, {**asst1, "reasoning_content": " "}, tool1],
        "C:real_replay": [user, {**asst1, "reasoning_content": r1}, tool1],
    }
    results = {}
    for name, msgs in conds.items():
        res = await complete(msgs, tools=TOOLS, stream=True)  # tools travel on continuations (OWUI does)
        if "error" in res:
            results[name] = res
            continue
        results[name] = {
            "finish": res["finish"],
            "reasoning_chars": len(res["reasoning"]),
            "reasoning": res["reasoning"][:200],
            "called": "get_weather" in res["content"] or res["finish"] == "tool_calls",
            "content": res["content"][:80],
        }
    return {"step1_reasoning_chars": len(r1), "tool_called_step1": tc1["function"]["name"], "conds": results}


async def main():
    print("=== authentic turn 1 ===", flush=True)
    turn1 = await run_turn1()
    print(f"reasoning={len(turn1['reasoning'])} chars, calls={[tc['function']['name'] for tc in turn1['tool_calls']]}")
    if not turn1["tool_calls"]:
        print("no tool call — abort", flush=True)
        return

    reps = 3
    print(f"\n=== single continuation, {reps} reps per condition (stream) ===", flush=True)
    for name, msgs in continuation_histories(turn1).items():
        res = await run_condition(name, msgs, reps)
        if "error" in res:
            print(f"{name}: ERROR {res['error']['body'][:200]}")
            continue
        print(f"{name}: median reasoning={res['reasoning_chars_median']} chars / {res['reasoning_deltas_median']} deltas | content={res['content_chars_median']} chars")
        print(f"    sample: {res.get('reasoning_sample','')!r}")

    print("\n=== 2-step chain (get_date -> get_weather) ===", flush=True)
    chain = await two_step_chain()
    print(f"step1 reasoning={chain.get('step1_reasoning_chars')} chars, called={chain.get('tool_called_step1')}")
    for name, res in chain.get("conds", {}).items():
        if "error" in res:
            print(f"{name}: ERROR {res}")
        else:
            print(f"{name}: finish={res['finish']} reasoning={res['reasoning_chars']} chars | {res['reasoning']!r}")
            print(f"    content: {res['content']!r}")


if __name__ == "__main__":
    asyncio.run(main())


async def thinking_disabled_test() -> dict:
    """Open WebUI sends thinking={'type':'disabled'} on tool-call continuations.
    Test: same continuation with vs without the disabled marker."""
    turn1 = await run_turn1()
    if not turn1["tool_calls"]:
        return {"error": "no tool call"}
    tool_msgs = []
    for tc in turn1["tool_calls"]:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        if name == "get_date":
            result = json.dumps({"date": "2026-09-02"})
        else:
            result = json.dumps({"temp": 28, "sky": "sunny", "city": args.get("city", "London")})
        tool_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    assistant = {"role": "assistant", "content": turn1["content"], "tool_calls": turn1["tool_calls"],
                 "reasoning_content": turn1["reasoning"]}
    user = {"role": "user", "content": "What's the weather like in London today?"}
    base_msgs = [user, {**assistant}, *tool_msgs]

    out = {}
    for label, extra in [
        ("thinking_disabled", {"thinking": {"type": "disabled"}}),
        ("no_thinking_field", {}),
    ]:
        body = {"model": MODEL, "messages": base_msgs, "stream": True, "temperature": 0.2, "tools": TOOLS, **extra}
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)) as client:
            r = await client.post(URL, json=body, headers=HEADERS)
            if r.status_code != 200:
                out[label] = {"error": r.status_code, "body": r.text[:200]}
                continue
            reasoning, content = [], []
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
            out[label] = {"reasoning_deltas": len(reasoning), "reasoning_chars": len("".join(reasoning)),
                          "content_chars": len("".join(content))}
    return out


async def _thinking_main():
    print("\n=== thinking:disabled vs no thinking field (pipe's _normalize_thinking_for_gateway) ===", flush=True)
    for rep in range(2):
        res = await thinking_disabled_test()
        if "error" in res:
            print(f"rep{rep}: ERROR {res}")
            continue
        for label, v in res.items():
            if "error" in v:
                print(f"rep{rep} {label}: ERROR {v.get('error')} {v.get('body','')[:200]}")
                continue
            print(f"rep{rep} {label}: reasoning_deltas={v['reasoning_deltas']} reasoning_chars={v['reasoning_chars']} content_chars={v['content_chars']}")


if __name__ == "__main__":
    asyncio.run(_thinking_main())
