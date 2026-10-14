"""The insight assistant.

The agent loop is simple: send the conversation to the model with the tools; when
the model asks for a tool, run it against the semantic layer and send back the
result; repeat until the model answers. Every tool result is kept as evidence,
and the final answer is validated against that evidence before it is shown.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from assistant.llm import LLMClient
from assistant.tools import BRIEF_TOOL, TOOLS, run_tool
from assistant.validation import ValidationResult, validate

MAX_STEPS = 10

SYSTEM_PROMPT = """You are the insight assistant for a pharmaceutical sales organisation in Australia and New Zealand.
You help sales managers understand their team's performance and decide what to do next.

Where your data comes from
- You can only see data through your tools. They read from the same governed semantic layer as the
  company's performance report, so your numbers and the report's numbers are the same.

Rules for numbers
- Never calculate a number yourself: no differences, sums, averages or growth rates. Quote values exactly
  as the tools return them, with the same rounding. If you need a number the tools do not give you, say so.
- Percentages are already in percent (58.3 means 58.3%). Market share change is in percentage points.

How to write
- Use plain-English metric names such as "sales vs target", "A-segment reach" or "field days". Never use field
  names like sales_vs_target_pct.
- Write money with a dollar sign and commas, exactly as given, e.g. $102,346.
- Do not work out new numbers yourself, including counts such as days not worked. Quote what the tools give.
- Never round or restate a number differently. A reach gap of -21.7 points is "21.7 points below target",
  not "a 22-point gap".
- Each insight tells one story about one territory and brand. Connect sales, coverage and field capacity when
  they explain each other (for example: fewer field days, so lower reach, so lower sales). Do not split one
  story into several insights.

Delivering your answer
- When asked for a brief, deliver it by calling submit_brief. Otherwise reply with plain text.
- In follow-up answers, get links by calling get_report_links, filtered to what you are describing, and write
  each one as a Markdown link, e.g. [Open Parramatta coverage in the report](link). Use the link exactly as
  returned: never add, remove or change a filter in it.
- Only ever call tools that are in your list.

Be efficient
- Each tool call costs time and money. Prefer get_team_overview, and when you need several tools,
  request them together in one step rather than one at a time.

Rules for evidence
- Every insight must come from a tool result, and must include the report_link from that same tool result,
  copied exactly. Never write, shorten or edit a link yourself.

How to interpret
- Judge results against targets and the product's priority. A P1 brand matters most. Over-servicing a P3 brand
  while P1 coverage is below target is a misallocation of field effort, not good activity.
- Separate what the data shows from what might explain it. Offer explanations as possibilities for the manager
  to check (for example, field days well below normal often mean leave, a vacancy or training). The manager
  knows context you do not.
- Focus on what someone can act on. Skip results that are on track unless they are a useful example to learn from.

Build data literacy
- When you use a metric, explain in plain English what it means and how to read it, using the governed
  definitions. Encourage the manager to open the report link and check the view themselves.

Stay in scope
- If a question cannot be answered from the tools (for example individual customers, forecasts, competitor
  plans or anything outside the data), say so plainly and suggest the report or the analytics team.

Write in plain Australian English. Be concise and specific."""

BRIEF_INSTRUCTIONS = """Write the monthly brief for {manager}, who manages the {region} region, for {month}.

1. Call get_team_overview once for the {region} region in {period}. It returns sales, customer coverage and
   field capacity together.
2. Choose the 3 or 4 stories that most need action, prioritising P1 brands. Add one positive example only if a
   territory is on track for sales and its P1 coverage is on track; otherwise leave it out.
3. Call get_report_links once, with one link per insight, filtered to that insight's territory and product
   (and segment, on the coverage view, if the story is about one segment).
4. Deliver the brief by calling submit_brief."""


@dataclass
class AgentResult:
    text: str
    evidence: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    validation: ValidationResult | None = None


class InsightAgent:
    def __init__(self, client: LLMClient):
        self.client = client

    def run(self, messages: list[dict], final_tool: dict | None = None) -> AgentResult:
        """Run the tool loop on a conversation. `messages` is updated in place.

        If `final_tool` is given, the model delivers its answer by calling it, and the
        tool's input (structured by its schema) becomes the answer.
        """
        tools = TOOLS + ([final_tool] if final_tool else [])
        evidence, calls = [], []
        for _ in range(MAX_STEPS):
            reply = self.client.create(SYSTEM_PROMPT, messages, tools)
            messages.append({"role": "assistant", "content": reply["content"]})
            tool_uses = [b for b in reply["content"] if b["type"] == "tool_use"]
            final = next((b for b in tool_uses if final_tool and b["name"] == final_tool["name"]), None)
            if final:
                text = json.dumps(final["input"], ensure_ascii=False)
                return AgentResult(text, evidence, calls, validate(text, evidence))
            if not tool_uses:
                text = "\n".join(b["text"] for b in reply["content"] if b["type"] == "text").strip()
                return AgentResult(text, evidence, calls, validate(text, evidence))

            results = []
            for use in tool_uses:
                try:
                    output = run_tool(use["name"], use["input"])
                    evidence.append(output)
                    calls.append({"tool": use["name"], "input": use["input"]})
                    content, is_error = json.dumps(output, default=str, separators=(",", ":")), False
                except Exception as exc:  # report the problem to the model rather than crash
                    content, is_error = f"Tool error: {exc}", True
                results.append({"type": "tool_result", "tool_use_id": use["id"], "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})

        raise RuntimeError("The assistant did not finish within the step limit.")

    def monthly_brief(self, manager: str, region: str, period: str) -> tuple[dict, AgentResult]:
        month = pd.Period(period).strftime("%B %Y")
        prompt = BRIEF_INSTRUCTIONS.format(manager=manager, region=region, month=month, period=period)
        messages = [{"role": "user", "content": prompt}]
        result = self.run(messages, final_tool=BRIEF_TOOL)
        return parse_brief(result.text), result


def parse_brief(text: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The assistant did not return a brief in the expected format.")
    brief = json.loads(cleaned[start:end + 1])
    if "insights" not in brief:
        raise ValueError("The brief has no insights.")
    return brief
