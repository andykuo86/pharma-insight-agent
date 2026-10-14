"""Tests for the model connections, without network calls.

Run:  pytest
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai.types.chat import ChatCompletionMessage

from assistant import llm
from assistant.agent import InsightAgent
from assistant.tools import TOOLS


def test_provider_detected_from_key(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    assert llm.configured_provider() == "gemini"
    assert llm.model_name("gemini") == "gemini-3.8-flash"
    monkeypatch.setenv("LLM_MODEL", "gemini-3-flash-preview")
    assert llm.model_name("gemini") == "gemini-3-flash-preview"


def test_tools_convert_to_openai_format():
    converted = {t["function"]["name"]: t["function"] for t in llm.to_openai_tools(TOOLS)}
    assert "parameters" not in converted["list_scope"]           # empty schema dropped
    assert converted["get_coverage"]["parameters"]["properties"]["segment"]["enum"] == ["A", "B", "C"]


class FakeOpenAIProvider:
    """Behaves like an OpenAI-format API: checks the request shape and replies in that format."""

    def __init__(self):
        self.requests = []

    def create(self, system, messages, tools):
        request = llm.to_openai_messages(system, messages)
        self.requests.append(request)
        if len(self.requests) == 1:
            reply = ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "get_sales_performance",
                             "arguments": json.dumps({"period": "2026-09", "territory": "T03", "product": "Cardiora"})},
                "extra_content": {"google": {"thought_signature": "abc"}},   # must be sent back unchanged
            }])
        else:
            result = json.loads(request[-1]["content"])
            reply = ChatCompletionMessage(role="assistant", content=f"Sales are at {result['total']['sales_vs_target_pct']}% of target. "
                                                                    f"See {result['report_link']}")
        return llm.from_openai_message(reply)


def test_agent_loop_works_with_openai_format_and_keeps_tool_call_data():
    provider = FakeOpenAIProvider()
    result = InsightAgent(provider).run([{"role": "user", "content": "How is Parramatta tracking on Cardiora?"}])
    second = provider.requests[1]
    assistant_turn = next(m for m in second if m["role"] == "assistant")
    tool_turn = next(m for m in second if m["role"] == "tool")
    assert assistant_turn["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "abc"}}
    assert tool_turn["tool_call_id"] == "call_1"
    assert "83.9%" in result.text and result.validation.ok
