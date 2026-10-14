"""Tests for the assistant, without calling a real model.

A scripted fake model asks for tools in a fixed order, then writes a brief from the
tool results. This exercises the real tool loop, semantic layer and validation.

Run:  pytest
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import semantic_layer as sl
from assistant.agent import InsightAgent, parse_brief
from assistant.tools import run_tool
from assistant.validation import validate


class ScriptedModel:
    """Asks for two tools, then writes a brief quoting what they returned."""

    def __init__(self, invent_a_number: bool = False):
        self.invent = invent_a_number

    def create(self, system, messages, tools):
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0:
            return {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_sales_performance",
                 "input": {"period": "2026-09", "territory": "T03", "product": "Cardiora"}}]}
        if step == 1:
            return {"stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "t2", "name": "get_coverage",
                 "input": {"period": "2026-09", "territory": "T03", "product": "Cardiora", "segment": "A"}}]}
        sales = json.loads(messages[2]["content"][0]["content"])
        cover = json.loads(messages[4]["content"][0]["content"])
        table = cover["table"]
        row = dict(zip(table["columns"], table["rows"][0]))
        reach = "61.7%" if self.invent else f"{row['reach_pct']}%"
        brief = {"headline": "Parramatta needs attention on Cardiora.", "insights": [{
            "title": "Parramatta, Cardiora",
            "what_is_happening": f"Sales are at {sales['total']['sales_vs_target_pct']}% of target and A-segment "
                                 f"reach is {reach} against a {row['reach_target_pct']}% target.",
            "why_it_matters": "Cardiora is the P1 launch brand.",
            "suggested_action": "Plan cover for A-segment customers while the new rep ramps up.",
            "coaching_question": "Which A-segment customers have you not seen since July?",
            "metric_id": "reach_pct",
            "report_link": cover["report_link"],
        }]}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "```json\n" + json.dumps(brief) + "\n```"}]}


def test_brief_runs_tools_and_passes_validation():
    brief, result = InsightAgent(ScriptedModel()).monthly_brief("Sarah Lim", "NSW & ACT", "2026-09")
    assert [c["tool"] for c in result.tool_calls] == ["get_sales_performance", "get_coverage"]
    assert result.validation.ok, result.validation.summary()
    assert "83.9%" in brief["insights"][0]["what_is_happening"]


def test_invented_number_is_flagged():
    _, result = InsightAgent(ScriptedModel(invent_a_number=True)).monthly_brief("Sarah Lim", "NSW & ACT", "2026-09")
    assert not result.validation.ok
    assert "61.7%" in result.validation.unmatched_numbers


def test_link_not_from_tools_is_flagged():
    evidence = [run_tool("get_sales_performance", {"period": "2026-09", "territory": "T03"})]
    check = validate("See http://localhost:8501/report?territory=T01", evidence)
    assert check.unknown_links


def test_tool_link_opens_report_with_same_filters():
    out = run_tool("get_coverage", {"period": "2026-09", "territory": "T16", "product": "Cardiora"})
    f = sl.Filters(period="2026-09", territory="T16", product="Cardiora")
    assert out["report_link"] == sl.report_link(f, "coverage", source="assistant")


def test_quoted_numbers_ignore_dates_and_ids():
    evidence = [{"x": 83.9}]
    check = validate("In September 2026, T03 was at 83.9% across 3 insights.", evidence)
    assert check.ok and check.numbers_checked == 1


def test_parse_brief_handles_code_fences():
    assert parse_brief('```json\n{"headline": "h", "insights": []}\n```')["headline"] == "h"


def test_assistant_page_explains_missing_api_key(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "assistant.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert "API key" in at.warning[0].value


def test_team_overview_returns_all_three_views_in_one_call():
    out = run_tool("get_team_overview", {"period": "2026-09", "region": "NSW & ACT"})
    f = sl.Filters(period="2026-09", region="NSW & ACT")
    assert out["sales"]["report_link"] == sl.report_link(f, "sales", source="assistant")
    assert out["coverage"]["report_link"] == sl.report_link(f, "coverage", source="assistant")
    assert out["field_capacity"]["report_link"] == sl.report_link(f, "field", source="assistant")
    table = out["sales"]["table"]
    ids = {row[table["columns"].index("territory_id")] for row in table["rows"]}
    assert ids == {"T01", "T02", "T03", "T04", "T05"}
    assert out["territory_names"]["T03"] == "Parramatta"


def test_region_overview_is_small_enough_for_free_tier_limits():
    out = run_tool("get_team_overview", {"period": "2026-09", "region": "NSW & ACT"})
    assert len(json.dumps(out, default=str, separators=(",", ":"))) < 10_000   # roughly 2,500 tokens


class SubmitToolModel:
    """Follows the new brief flow: overview, then precise links, then submit_brief."""

    def create(self, system, messages, tools):
        step = sum(1 for m in messages if m["role"] == "assistant")
        if step == 0:
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "t1", "name": "get_team_overview",
                                                            "input": {"period": "2026-09", "region": "NSW & ACT"}}]}
        if step == 1:
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "t2", "name": "get_report_links",
                    "input": {"links": [{"period": "2026-09", "territory": "T03", "product": "Cardiora",
                                         "segment": "A", "view": "coverage"}]}}]}
        overview = json.loads(messages[2]["content"][0]["content"])
        links = json.loads(messages[4]["content"][0]["content"])["links"]
        cov = overview["coverage"]["table"]
        rows = [dict(zip(cov["columns"], r)) for r in cov["rows"]]
        a = next(r for r in rows if r["territory_id"] == "T03" and r["product"] == "Cardiora" and r["segment"] == "A")
        brief = {"headline": "Parramatta needs attention.", "insights": [{
            "title": "Parramatta, Cardiora",
            "what_is_happening": f"A-segment reach is {a['reach_pct']}%, {abs(a['reach_gap_pts'])} points below target.",
            "why_it_matters": "P1 launch brand.", "suggested_action": "Cover A-segment customers.",
            "coaching_question": "Which A-segment customers have you not seen?", "metric_id": "reach_pct",
            "report_link": links[0]["report_link"]}]}
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "t3", "name": "submit_brief", "input": brief}]}


def test_brief_delivered_through_submit_tool_with_precise_link():
    brief, result = InsightAgent(SubmitToolModel()).monthly_brief("Sarah Lim", "NSW & ACT", "2026-09")
    expected = sl.report_link(sl.Filters(period="2026-09", territory="T03", product="Cardiora", segment="A"),
                              "coverage", source="assistant")
    assert brief["insights"][0]["report_link"] == expected
    assert result.validation.ok, result.validation.summary()


def test_rounded_number_is_flagged():
    check = validate("Reach is 21.7 points below target, a 22‑point gap.", [{"reach_gap_pts": -21.7}])
    assert check.unmatched_numbers == ["22‑point"]


def test_comma_grouped_values_are_checked():
    assert validate("Sales were 102,346 against target.", [{"sales_value": 102346}]).ok
    assert not validate("Sales were 102,999 against target.", [{"sales_value": 102346}]).ok


def test_invalid_final_tool_call_is_recovered_as_text():
    from assistant.llm import recover_invalid_tool_call
    body = {"message": "Tool call validation failed", "code": "tool_use_failed",
            "failed_generation": '{"name": "json", "arguments": {"headline": "H", "insights": []}}'}
    recovered = recover_invalid_tool_call(body, {"get_team_overview", "submit_brief"})
    assert parse_brief(recovered["content"][0]["text"])["headline"] == "H"
    assert recover_invalid_tool_call({"code": "other"}, set()) is None


def test_unusual_hyphens_in_model_output_do_not_break_validation():
    check = validate("Field days were 60.0%, compared with 90‑95% elsewhere.", [{"a": 60.0}, {"b": 95.0}])
    assert check.ok


def test_report_page_is_reachable_at_its_own_path():
    """Regression test: the default page is always served at the root, so the report must not be default."""
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    report_line = next(line for line in source.splitlines() if "views/report.py" in line)
    assert 'url_path="report"' in report_line and "default=True" not in report_line
    assert sl.REPORT_BASE_URL.endswith("/report")


def test_links_wrapped_in_unusual_brackets_are_still_checked():
    link = sl.report_link(sl.Filters(period="2026-09", region="NSW & ACT"), "sales", source="assistant")
    edited = link.replace("&view", "&product=Cardiora&view")
    assert validate(f"【report_link: {link}】", [{"report_link": link}]).ok
    assert not validate(f"【report_link: {edited}】", [{"report_link": link}]).ok


def test_app_opens_on_the_assistant(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Insight assistant"


def test_dollar_signs_are_escaped_once_for_display():
    source = (Path(__file__).resolve().parents[1] / "views" / "assistant.py").read_text()
    ns = {}
    exec(source[source.index("def md"):source.index("def show_check")], ns)
    assert ns["md"]("Sales $102,346") == "Sales \\$102,346"
