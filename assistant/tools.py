"""Tools the assistant can call.

Every tool reads from the semantic layer, the same source as the report, and returns:
  - the rows it found, exactly as the semantic layer calculated and rounded them
  - the report link for the same filters, so any insight can be checked in one click
  - plain-English definitions of the metrics involved, to help the reader learn

The assistant never sees raw data and never calculates anything itself.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

import semantic_layer as sl

_FILTER_PROPERTIES = {
    "period": {"type": "string", "description": "Month as YYYY-MM. Defaults to the latest month."},
    "region": {"type": "string", "description": "Region name, e.g. 'NSW & ACT'."},
    "territory": {"type": "string", "description": "Territory ID, e.g. 'T03'."},
    "product": {"type": "string", "description": "Product name: Cardiora, Neurava or Dermalyn."},
    "segment": {"type": "string", "enum": ["A", "B", "C"], "description": "Customer segment."},
}

_LINK_REQUEST = {
    "type": "object",
    "properties": {
        "period": {"type": "string"},
        "territory": {"type": "string"},
        "product": {"type": "string"},
        "segment": {"type": "string", "enum": ["A", "B", "C"]},
        "view": {"type": "string", "enum": ["sales", "coverage", "field"]},
    },
    "required": ["view"],
}

TOOLS: list[dict] = [
    {
        "name": "list_scope",
        "description": "List the months, regions, territories (with rep and manager), and products "
                       "(with priority and strategy) available in the data. Use this first if unsure of names or IDs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_team_overview",
        "description": "Everything needed for a monthly brief in one call: sales performance, customer coverage "
                       "and field capacity for every territory in scope, each with its report link and "
                       "how-to-read notes. Use this first for a region or territory, then other tools only for more detail.",
        "input_schema": {"type": "object", "properties": _FILTER_PROPERTIES},
    },
    {
        "name": "get_sales_performance",
        "description": "Sales, target, gap to target, sales vs target %, market share % and market share change "
                       "for each territory and product in scope, plus a total for the scope.",
        "input_schema": {"type": "object", "properties": _FILTER_PROPERTIES},
    },
    {
        "name": "get_coverage",
        "description": "Reach % and frequency against target for each territory, product and customer segment in scope, "
                       "with status. Use to see whether the right customers are being seen often enough for each brand's priority.",
        "input_schema": {"type": "object", "properties": _FILTER_PROPERTIES},
    },
    {
        "name": "get_field_capacity",
        "description": "Field days worked and call rate for each territory in scope (across all products). "
                       "Use to check whether a drop in activity is about time in the field.",
        "input_schema": {"type": "object", "properties": _FILTER_PROPERTIES},
    },
    {
        "name": "get_trend",
        "description": "Monthly sales vs target % and market share % for the last 12 months up to the selected month.",
        "input_schema": {"type": "object", "properties": _FILTER_PROPERTIES},
    },
    {
        "name": "get_report_links",
        "description": "Report links filtered to exactly what an insight is about, e.g. one territory, product and "
                       "segment on the coverage view. Request all the links you need in one call.",
        "input_schema": {"type": "object", "properties": {"links": {"type": "array", "items": _LINK_REQUEST}},
                         "required": ["links"]},
    },
    {
        "name": "get_metric_definition",
        "description": "The governed definition of a metric: what it means, how it is calculated and how to read it.",
        "input_schema": {
            "type": "object",
            "properties": {"metric_id": {"type": "string", "description": "e.g. reach_pct, frequency, market_share_change_pts"}},
            "required": ["metric_id"],
        },
    },
]


def _records(df: pd.DataFrame) -> list[dict]:
    rows = df.to_dict(orient="records")
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]


# Columns the model does not need: names are given once in territory_names, counts are not interpreted.
_DROP = ["territory_name", "region", "manager", "customers_in_segment", "customers_reached", "calls"]


def _table(df: pd.DataFrame) -> dict:
    """Compact table: column names once, then rows as lists. Far fewer tokens than one dict per row."""
    df = df.drop(columns=[c for c in _DROP if c in df.columns])
    rows = _records(df)
    columns = list(df.columns)
    return {"columns": columns, "rows": [[r[c] for c in columns] for r in rows]}


def _names(f: "sl.Filters") -> dict:
    terr = sl.territories()
    return dict(zip(terr["territory_id"], terr["territory_name"])) if not f.territory else {
        f.territory: terr.set_index("territory_id").loc[f.territory, "territory_name"]}


def _how_to_read(*metric_ids: str) -> dict[str, str]:
    return {m: sl.definition(m)["how_to_read"] for m in metric_ids}


def run_tool(name: str, args: dict[str, Any]) -> dict:
    """Run a tool and return a JSON-serialisable result."""
    if name == "list_scope":
        return {
            "months": sl.periods(),
            "latest_month": sl.latest_period(),
            "territories": _records(sl.territories()),
            "products": _records(sl.products()),
        }

    if name == "get_metric_definition":
        return sl.definition(args["metric_id"])

    if name == "get_report_links":
        out = []
        for req in args.get("links", []):
            f = sl.Filters.from_params({k: v for k, v in req.items() if v and k != "view"})
            out.append({"for": sl.describe(f), "view": req["view"],
                        "report_link": sl.report_link(f, req["view"], source="assistant")})
        return {"links": out}

    filters = sl.Filters.from_params({k: v for k, v in args.items() if v})
    scope = {"filters_used": filters.to_params(), "scope_description": sl.describe(filters)}
    if filters.region or filters.territory:
        ids = set(sl._territory_ids(filters))
        scope["territory_names"] = {k: v for k, v in _names(filters).items() if k in ids}
    else:
        scope["territory_names"] = _names(filters)

    if name == "get_team_overview":
        strip = lambda d: {k: v for k, v in d.items() if k not in scope}   # don't repeat the scope three times
        return scope | {
            "sales": strip(run_tool("get_sales_performance", args)),
            "coverage": strip(run_tool("get_coverage", args)),
            "field_capacity": strip(run_tool("get_field_capacity", args)),
        }
    if name == "get_sales_performance":
        return scope | {
            "report_link": sl.report_link(filters, "sales", source="assistant"),
            "total": _records(sl.sales_performance(filters, by=()))[0],
            "table": _table(sl.sales_performance(filters)),
            "how_to_read": _how_to_read("sales_vs_target_pct", "market_share_change_pts"),
        }
    if name == "get_coverage":
        return scope | {
            "report_link": sl.report_link(filters, "coverage", source="assistant"),
            "table": _table(sl.coverage(filters)),
            "how_to_read": _how_to_read("reach_pct", "frequency"),
        }
    if name == "get_field_capacity":
        return scope | {
            "report_link": sl.report_link(filters, "field", source="assistant"),
            "table": _table(sl.field_capacity(filters)),
            "how_to_read": _how_to_read("field_day_pct", "call_rate"),
        }
    if name == "get_trend":
        return scope | {
            "report_link": sl.report_link(filters, "sales", source="assistant"),
            "table": _table(sl.trend(filters)),
        }
    raise ValueError(f"Unknown tool '{name}'")


# The brief is delivered through a tool, so its structure is enforced by the schema.
BRIEF_TOOL: dict = {
    "name": "submit_brief",
    "description": "Submit the finished monthly brief. Call this once, at the end, instead of replying with text.",
    "input_schema": {
        "type": "object",
        "required": ["headline", "insights"],
        "properties": {
            "headline": {"type": "string", "description": "One sentence summary of the month for this region."},
            "insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["title", "what_is_happening", "why_it_matters", "suggested_action",
                                 "coaching_question", "metric_id", "report_link"],
                    "properties": {
                        "title": {"type": "string", "description": "Short title naming the territory and brand."},
                        "what_is_happening": {"type": "string", "description": "The facts, numbers quoted exactly as the tools returned them."},
                        "why_it_matters": {"type": "string", "description": "Why it matters given targets and the brand's priority."},
                        "suggested_action": {"type": "string"},
                        "coaching_question": {"type": "string", "description": "One question for the rep's next coaching conversation."},
                        "metric_id": {"type": "string", "description": "The main governed metric, e.g. reach_pct."},
                        "report_link": {"type": "string", "description": "A link from get_report_links, filtered to this insight."},
                    },
                },
            },
        },
    },
}
