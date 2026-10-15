"""Simulated semantic layer.

In a real business this logic lives in the Power BI semantic model, a Tableau data
source or a Foundry ontology. Here it is one Python module, and it is the ONLY place
metrics are calculated. The performance report and the AI assistant both call it,
so they cannot show different numbers.

Two rules keep it trustworthy:
  1. Ratios are always SUM(numerator) / SUM(denominator) across the filtered rows,
     never an average of ratios. That matches how a governed measure behaves.
  2. Rounding happens here, once. The report and the assistant show the same
     rounded values, so a number quoted by the assistant matches the report exactly.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import urlencode

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
REPORT_BASE_URL = os.environ.get("REPORT_BASE_URL", "http://localhost:8501/report")
REPORT_VIEWS = ("sales", "coverage", "field", "definitions")

# Status rules. They are also written in plain English in metric_definitions.csv,
# so the assistant can explain them.
SALES_ON_TRACK, SALES_WATCH = 98.0, 90.0
REACH_WATCH_GAP_PTS = 10.0
FREQ_UNDER, FREQ_OVER = 0.8, 1.5


# --------------------------------------------------------------------------- data
@lru_cache
def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


@lru_cache
def metadata() -> dict:
    return json.loads((DATA_DIR / "metadata.json").read_text())


def periods() -> list[str]:
    return sorted(_table("fact_sales")["month"].unique().tolist())


def latest_period() -> str:
    return periods()[-1]


def territories() -> pd.DataFrame:
    return _table("dim_territory").copy()


def products() -> pd.DataFrame:
    return _table("dim_product").copy()


def definitions() -> pd.DataFrame:
    return _table("metric_definitions").copy()


def definition(metric_id: str) -> dict:
    rows = definitions()
    match = rows[rows["metric_id"] == metric_id]
    if match.empty:
        raise KeyError(f"No governed definition for metric '{metric_id}'")
    return match.iloc[0].to_dict()


# ------------------------------------------------------------------------ filters
@dataclass(frozen=True)
class Filters:
    """One filter object drives both the data query and the report link."""

    period: str
    region: str | None = None
    territory: str | None = None   # territory_id, e.g. "T03"
    product: str | None = None
    segment: str | None = None     # "A", "B" or "C"

    def to_params(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}

    @classmethod
    def from_params(cls, params: Mapping[str, str]) -> "Filters":
        """Build filters from URL query parameters, ignoring any invalid values."""
        terr = territories()
        valid = {
            "period": set(periods()),
            "region": set(terr["region"]),
            "territory": set(terr["territory_id"]),
            "product": set(products()["product"]),
            "segment": {"A", "B", "C"},
        }
        clean = {k: params.get(k) for k in valid if params.get(k) in valid[k]}
        clean.setdefault("period", latest_period())
        return cls(**clean)


def _dashboard_link(base: str, filters: Filters, view: str) -> str:
    """Link into the hosted dashboard: <url>#<tab>_t-T03_p-Cardiora_s-A_m-2026-09.

    The tab and the filters travel as one name made of letters, digits, hyphens and
    underscores, which the dashboard reads when it opens. A region is sent as its id
    (letters and digits, with hyphens), e.g. "NSW & ACT" becomes "NSW-ACT".
    """
    region = re.sub(r"[^A-Za-z0-9]+", "-", filters.region) if filters.region else None
    values = [("r", region), ("t", filters.territory), ("p", filters.product), ("s", filters.segment), ("m", filters.period)]
    return base.rstrip("/") + "#" + "_".join([view] + [f"{key}-{value}" for key, value in values if value])


def report_link(filters: Filters, view: str = "sales", source: str | None = None) -> str:
    """Link that opens the report with exactly these filters pre-selected.

    Opens the hosted dashboard when REPORT_DASHBOARD_URL is set, otherwise the local
    Streamlit report.
    """
    if view not in REPORT_VIEWS:
        raise ValueError(f"Unknown report view '{view}'")
    dashboard = os.environ.get("REPORT_DASHBOARD_URL")
    if dashboard:
        return _dashboard_link(dashboard, filters, view)
    params = filters.to_params() | {"view": view}
    if source:
        params["from"] = source
    return f"{REPORT_BASE_URL}?{urlencode(params)}"


def describe(filters: Filters) -> str:
    """Plain-English summary of the filter context, e.g. for headings."""
    terr = territories().set_index("territory_id")
    parts = []
    if filters.territory:
        parts.append(f"{terr.loc[filters.territory, 'territory_name']} ({filters.territory})")
    elif filters.region:
        parts.append(f"{filters.region} region")
    else:
        parts.append("ANZ")
    if filters.product:
        parts.append(filters.product)
    if filters.segment:
        parts.append(f"{filters.segment}-segment customers")
    month = pd.Period(filters.period).strftime("%B %Y")
    return ", ".join(parts) + f", {month}"


def _territory_ids(f: Filters) -> list[str]:
    terr = territories()
    if f.territory:
        return [f.territory]
    if f.region:
        return terr.loc[terr["region"] == f.region, "territory_id"].tolist()
    return terr["territory_id"].tolist()


def _scope(df: pd.DataFrame, f: Filters, *, use_segment: bool = False) -> pd.DataFrame:
    out = df[df["territory_id"].isin(_territory_ids(f))]
    if f.product and "product" in out:
        out = out[out["product"] == f.product]
    if use_segment and f.segment:
        out = out[out["segment"] == f.segment]
    return out


def _prior_months(period: str, n: int = 3) -> list[str]:
    p = pd.Period(period)
    return [str(p - k) for k in range(1, n + 1)]


def _pct(num: float, den: float) -> float | None:
    return round(100 * num / den, 1) if den else None


def _sales_status(pct: float | None) -> str:
    if pct is None:
        return "No data"
    if pct >= SALES_ON_TRACK:
        return "On track"
    return "Watch" if pct >= SALES_WATCH else "Off track"


# ----------------------------------------------------------------------- measures
def sales_performance(f: Filters, by: tuple[str, ...] = ("territory_id", "product")) -> pd.DataFrame:
    """Sales, target, sales vs target and market share for the selected month.

    `by` sets the grain: ("territory_id", "product") for the detail table, () for a total.
    """
    sales = _scope(_table("fact_sales"), f)
    cur = sales[sales["month"] == f.period]
    prior = sales[sales["month"].isin(_prior_months(f.period))]
    keys = list(by)

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["sales_value", "target_value", "market_value"]
        return df.groupby(keys)[cols].sum().reset_index() if keys else df[cols].sum().to_frame().T

    now, before = agg(cur), agg(prior).rename(columns={"sales_value": "prior_sales", "market_value": "prior_market"})
    out = now.merge(before[keys + ["prior_sales", "prior_market"]], on=keys, how="left") if keys else now.join(before[["prior_sales", "prior_market"]])

    out["sales_vs_target_pct"] = [_pct(s, t) for s, t in zip(out.sales_value, out.target_value)]
    out["market_share_pct"] = [_pct(s, m) for s, m in zip(out.sales_value, out.market_value)]
    prior_share = [_pct(s, m) for s, m in zip(out.prior_sales, out.prior_market)]
    out["market_share_change_pts"] = [
        round(now_s - was, 1) if now_s is not None and was is not None else None
        for now_s, was in zip(out.market_share_pct, prior_share)
    ]
    out["sales_gap_value"] = (out["sales_value"] - out["target_value"]).round().astype(int)
    out["status"] = out["sales_vs_target_pct"].map(_sales_status)
    out = out.drop(columns=["prior_sales", "prior_market", "market_value"])
    out[["sales_value", "target_value"]] = out[["sales_value", "target_value"]].astype(int)
    return _label(out)


def coverage(f: Filters, by: tuple[str, ...] = ("territory_id", "product", "segment")) -> pd.DataFrame:
    """Reach and frequency against target for the selected month."""
    act = _scope(_table("fact_activity"), f, use_segment=True)
    act = act[act["month"] == f.period]
    keys = list(by)
    cols = ["customers_in_segment", "customers_reached", "calls"]
    out = act.groupby(keys)[cols].sum().reset_index()

    out["reach_pct"] = [_pct(r, n) for r, n in zip(out.customers_reached, out.customers_in_segment)]
    out["frequency"] = [round(c / r, 1) if r else None for c, r in zip(out.calls, out.customers_reached)]

    if {"product", "segment"} <= set(keys):
        tgt = _table("targets")
        out = out.merge(tgt, on=["product", "segment"], how="left")
        out["reach_target_pct"] = (out["reach_target"] * 100).round(1)
        out["frequency_target"] = out["frequency_target"].round(1)
        out["reach_gap_pts"] = [
            round(r - t, 1) if r is not None else None for r, t in zip(out.reach_pct, out.reach_target_pct)
        ]
        out["reach_status"] = [
            "No data" if r is None else "On track" if r >= t else "Watch" if r >= t - REACH_WATCH_GAP_PTS else "Off track"
            for r, t in zip(out.reach_pct, out.reach_target_pct)
        ]
        out["frequency_status"] = [
            "No data" if fr is None else "Under-servicing" if fr < FREQ_UNDER * t
            else "Over-servicing" if fr > FREQ_OVER * t else "On target"
            for fr, t in zip(out.frequency, out.frequency_target)
        ]
        out = out.drop(columns=["reach_target"])
    return _label(out)


def field_capacity(f: Filters) -> pd.DataFrame:
    """Field days and call rate per territory for the selected month (all products)."""
    days = _table("fact_field_days")
    days = days[(days["month"] == f.period) & days["territory_id"].isin(_territory_ids(f))]
    act = _table("fact_activity")
    act = act[(act["month"] == f.period) & act["territory_id"].isin(_territory_ids(f))]
    calls = act.groupby("territory_id")["calls"].sum().reset_index()
    out = days.merge(calls, on="territory_id", how="left").drop(columns="month")
    out["field_day_pct"] = [_pct(d, a) for d, a in zip(out.field_days, out.available_days)]
    out["call_rate"] = [round(c / d, 1) if d else None for c, d in zip(out.calls, out.field_days)]
    return _label(out)


def trend(f: Filters, months: int = 12) -> pd.DataFrame:
    """Monthly sales vs target and market share for the filter scope, ending at the selected month."""
    end = pd.Period(f.period)
    window = [str(end - k) for k in range(months - 1, -1, -1)]
    sales = _scope(_table("fact_sales"), f)
    sales = sales[sales["month"].isin(window)]
    out = sales.groupby("month")[["sales_value", "target_value", "market_value"]].sum().reset_index()
    out["sales_vs_target_pct"] = [_pct(s, t) for s, t in zip(out.sales_value, out.target_value)]
    out["market_share_pct"] = [_pct(s, m) for s, m in zip(out.sales_value, out.market_value)]
    return out.drop(columns="market_value")


def _label(df: pd.DataFrame) -> pd.DataFrame:
    """Add territory names and product priority so tables are readable."""
    if "territory_id" in df:
        df = df.merge(territories()[["territory_id", "territory_name", "region", "manager"]], on="territory_id", how="left")
    if "product" in df:
        df = df.merge(products()[["product", "priority"]], on="product", how="left")
    return df
