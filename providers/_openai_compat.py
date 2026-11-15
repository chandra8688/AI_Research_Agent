"""Shared conversion helpers for OpenAI-compatible chat APIs (Groq, OpenRouter).

The agent keeps its conversation in a provider-neutral format:
  {"role": "user", "content": str}
  {"role": "user", "function_responses": [{"name", "response": {"result"}, "id"?}]}
  {"role": "model", "raw_message": <provider-native assistant message>}
  {"role": "model" | "assistant", "content": str}   # plain prior answers
"""
import copy
from typing import Any


def _lowercase_types(schema: Any) -> None:
    if isinstance(schema, dict):
        if "type" in schema and isinstance(schema["type"], str):
            schema["type"] = schema["type"].lower()
        for v in schema.values():
            _lowercase_types(v)
    elif isinstance(schema, list):
        for item in schema:
            _lowercase_types(item)


def to_openai_tools(tools: list[dict]) -> list[dict]:
    converted = []
    for t in tools or []:
        params = copy.deepcopy(t["parameters"])
        _lowercase_types(params)
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": params,
            },
        })
    return converted


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Converts neutral messages to OpenAI chat format.

    Every tool message must reference the id of the tool call it answers. Responses
    are matched to the preceding assistant tool_calls by explicit id when present,
    otherwise by position, so parallel tool calls each get the correct id.
    """
    converted: list[dict] = []
    last_tool_call_ids: list[str] = []

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            if "function_responses" in msg:
                for i, resp in enumerate(msg["function_responses"]):
                    call_id = resp.get("id")
                    if not call_id:
                        call_id = last_tool_call_ids[i] if i < len(last_tool_call_ids) else "unknown_id"
                    converted.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": resp["name"],
                        "content": str(resp["response"].get("result", "")),
                    })
            else:
                converted.append({"role": "user", "content": msg.get("content", "")})
        elif role in ("model", "assistant"):
            raw = msg.get("raw_message")
            if isinstance(raw, dict):
                raw = dict(raw)
                raw["role"] = "assistant"
                converted.append(raw)
                last_tool_call_ids = [tc.get("id") for tc in raw.get("tool_calls") or []]
            elif msg.get("content"):
                converted.append({"role": "assistant", "content": msg["content"]})

    return converted
