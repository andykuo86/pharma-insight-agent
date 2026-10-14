"""Connection to the language model.

The assistant works with any model that can call tools. Two connection classes cover
most options:

  AnthropicClient         Claude, via the Anthropic API.
  OpenAICompatibleClient  Any provider that accepts the OpenAI request format:
                          Google Gemini (free tier), Groq, OpenRouter, or a company's
                          own enterprise gateway.

Both return the same shape, so the rest of the app does not care which is used:
    {"stop_reason": "tool_use" | "end_turn",
     "content": [{"type": "text", "text": ...},
                 {"type": "tool_use", "id": ..., "name": ..., "input": {...}}]}

Choosing a provider (environment variables):
  LLM_PROVIDER   anthropic | gemini | groq | openrouter   (optional: auto-detected from the key you set)
  LLM_MODEL      optional model override
  API key        ANTHROPIC_API_KEY | GEMINI_API_KEY | GROQ_API_KEY | OPENROUTER_API_KEY
"""
from __future__ import annotations

import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict: ...


# provider: (base URL, API key variable, default model)
PROVIDERS = {
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.8-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", None),  # set LLM_MODEL
}


def configured_provider() -> str | None:
    """The provider to use: LLM_PROVIDER if set, otherwise the first one with an API key."""
    chosen = os.environ.get("LLM_PROVIDER", "").lower()
    if chosen in PROVIDERS:
        return chosen if os.environ.get(PROVIDERS[chosen][1]) else None
    return next((p for p, (_, key, _) in PROVIDERS.items() if os.environ.get(key)), None)


def api_key_configured() -> bool:
    return configured_provider() is not None


def model_name(provider: str) -> str | None:
    override = os.environ.get("LLM_MODEL")
    if not override and provider == "anthropic":
        override = os.environ.get("ANTHROPIC_MODEL")   # kept for compatibility
    return override or PROVIDERS[provider][2]


def get_client() -> "LLMClient":
    provider = configured_provider()
    if provider is None:
        raise RuntimeError("No API key found. Set GEMINI_API_KEY (free), ANTHROPIC_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY.")
    model = model_name(provider)
    if not model:
        raise RuntimeError(f"Set LLM_MODEL to choose a {provider} model.")
    if provider == "anthropic":
        return AnthropicClient(model)
    base_url, key_var, _ = PROVIDERS[provider]
    return OpenAICompatibleClient(base_url, os.environ[key_var], model, label=provider)


# ------------------------------------------------------------------- Anthropic
class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 4000):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model, self.max_tokens, self.label = model, max_tokens, "anthropic"

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, tools=tools, messages=messages,
        )
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        return {"stop_reason": response.stop_reason, "content": content}


# ----------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleClient:
    def __init__(self, base_url: str | None, api_key: str, model: str, label: str = "openai-compatible"):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)   # waits and retries on rate limits
        self.model, self.label = model, label

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        from openai import BadRequestError

        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=to_openai_messages(system, messages), tools=to_openai_tools(tools),
            )
        except BadRequestError as exc:
            recovered = recover_invalid_tool_call(exc.body, {t["name"] for t in tools})
            if recovered is None:
                raise
            return recovered
        return from_openai_message(response.choices[0].message)


def recover_invalid_tool_call(body, tool_names: set[str]) -> dict | None:
    """Some open models deliver their final answer as a call to a tool that does not exist
    (e.g. a tool named "json"). The provider rejects it but returns what the model wrote.
    Treat that as the model's text answer instead of failing."""
    error = body.get("error", body) if isinstance(body, dict) else {}
    if error.get("code") != "tool_use_failed" or not error.get("failed_generation"):
        return None
    try:
        attempt = json.loads(error["failed_generation"])
    except (TypeError, ValueError):
        return None
    if not isinstance(attempt, dict) or attempt.get("name") in tool_names:
        return None
    arguments = attempt.get("arguments", "")
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = {"name": t["name"], "description": t["description"]}
        if t["input_schema"].get("properties"):          # some providers reject empty parameter lists
            fn["parameters"] = t["input_schema"]
        out.append({"type": "function", "function": fn})
    return out


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert the app's conversation (Anthropic-style blocks) to the OpenAI format."""
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
        elif m["role"] == "user":                         # tool results
            for block in content:
                if block["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})
                elif block["type"] == "text":
                    out.append({"role": "user", "content": block["text"]})
        else:                                             # assistant turn, possibly with tool calls
            text = "\n".join(b["text"] for b in content if b["type"] == "text") or None
            calls = [
                b.get("_raw") or {"id": b["id"], "type": "function",
                                  "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content if b["type"] == "tool_use"
            ]
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


def from_openai_message(message) -> dict:
    """Convert an OpenAI-format reply to the app's shape.

    The original tool call is kept in `_raw` and sent back unchanged on the next turn,
    because some models (e.g. Gemini 3) attach extra data to tool calls that must be returned.
    """
    content = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for call in message.tool_calls or []:
        raw = call.model_dump(exclude_none=True) if hasattr(call, "model_dump") else dict(call)
        content.append({"type": "tool_use", "id": call.id, "name": call.function.name,
                        "input": json.loads(call.function.arguments or "{}"), "_raw": raw})
    return {"stop_reason": "tool_use" if message.tool_calls else "end_turn", "content": content}
