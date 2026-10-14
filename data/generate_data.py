"""Generate the synthetic dataset for the demo.

Everything here is invented: the company, brands, territories, people and numbers.
No real sales, customer or healthcare professional data is used.

The data stands in for the base tables that sit underneath a reporting tool's
semantic model. Four scenarios are deliberately built in so the AI assistant
has real, actionable stories to find:

  T03 Parramatta       Cardiora behind target: rep vacancy in Jul-Aug cut field days
                       and A-segment reach.
  T07 Melbourne South  Cardiora near target, but market share falling since a
                       competitor launch in June (market growing, share dropping).
  T16 Auckland Central Field effort misallocated: Dermalyn (P3) over-called while
                       Cardiora (P1) A-segment reach is well below target.
  T13 Perth North      Strong performer: above target with high A-segment reach.

Run:  python data/generate_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

MONTHS = pd.period_range("2025-10", "2026-09", freq="M").astype(str).tolist()
AVAILABLE_DAYS = {m: 20 for m in MONTHS} | {"2025-12": 15, "2026-01": 16}

territories = pd.DataFrame([
    ("T01", "Sydney CBD", "NSW", "NSW & ACT", "Olivia Grant", "Sarah Lim"),
    ("T02", "North Shore", "NSW", "NSW & ACT", "Daniel Wu", "Sarah Lim"),
    ("T03", "Parramatta", "NSW", "NSW & ACT", "Priya Nair", "Sarah Lim"),
    ("T04", "Newcastle", "NSW", "NSW & ACT", "Tom Hayes", "Sarah Lim"),
    ("T05", "Canberra", "ACT", "NSW & ACT", "Grace Okafor", "Sarah Lim"),
    ("T06", "Melbourne North", "VIC", "VIC & TAS", "Liam Murphy", "James Patel"),
    ("T07", "Melbourne South", "VIC", "VIC & TAS", "Chloe Martin", "James Patel"),
    ("T08", "Geelong & West", "VIC", "VIC & TAS", "Ben Russo", "James Patel"),
    ("T09", "Hobart", "TAS", "VIC & TAS", "Hannah Lee", "James Patel"),
    ("T10", "Brisbane North", "QLD", "QLD", "Jack Wilson", "Mia Thompson"),
    ("T11", "Brisbane South", "QLD", "QLD", "Zara Ahmed", "Mia Thompson"),
    ("T12", "Gold Coast", "QLD", "QLD", "Noah Clarke", "Mia Thompson"),
    ("T13", "Perth North", "WA", "WA & SA", "Ella Fraser", "Ravi Menon"),
    ("T14", "Perth South", "WA", "WA & SA", "Lucas Brown", "Ravi Menon"),
    ("T15", "Adelaide", "SA", "WA & SA", "Isla Turner", "Ravi Menon"),
    ("T16", "Auckland Central", "NZ", "New Zealand", "Mason Tane", "Kiri Walker"),
    ("T17", "Wellington", "NZ", "New Zealand", "Ruby Chen", "Kiri Walker"),
    ("T18", "Christchurch", "NZ", "New Zealand", "Finn O'Brien", "Kiri Walker"),
], columns=["territory_id", "territory_name", "state", "region", "rep", "manager"])

products = pd.DataFrame([
    ("Cardiora", "P1", "Cardiometabolic",
     "Launch-phase growth brand. Prioritise A-segment cardiologists and high-volume GPs."),
    ("Neurava", "P2", "Neurology",
     "Established brand. Defend share and keep frequency with existing prescribers."),
    ("Dermalyn", "P3", "Dermatology",
     "Mature brand. Maintain with minimal field effort; digital-first engagement."),
], columns=["product", "priority", "therapeutic_area", "strategy_note"])

# Targets the business has set, by product priority and customer segment.
targets = pd.DataFrame([
    (p, seg, reach, freq)
    for p, rows in {
        "Cardiora": [("A", 0.80, 3.0), ("B", 0.60, 2.0), ("C", 0.30, 1.0)],
        "Neurava":  [("A", 0.70, 2.0), ("B", 0.50, 1.5), ("C", 0.20, 1.0)],
        "Dermalyn": [("A", 0.40, 1.0), ("B", 0.25, 0.5), ("C", 0.10, 0.5)],
    }.items()
    for seg, reach, freq in rows
], columns=["product", "segment", "reach_target", "frequency_target"])

# Market size per territory per month (AUD), starting share, monthly share drift.
PRODUCT_MODEL = {
    "Cardiora": dict(market=(400_000, 900_000), market_growth=0.015, share=0.12, share_drift=0.004),
    "Neurava":  dict(market=(300_000, 600_000), market_growth=0.000, share=0.30, share_drift=0.000),
    "Dermalyn": dict(market=(150_000, 300_000), market_growth=-0.003, share=0.22, share_drift=-0.0005),
}

sales_rows, activity_rows, days_rows = [], [], []

for t in territories.territory_id:
    universe = {"A": int(rng.integers(10, 16)), "B": int(rng.integers(25, 36)), "C": int(rng.integers(50, 71))}
    for i, m in enumerate(MONTHS):
        avail = AVAILABLE_DAYS[m]
        field_days = avail - int(rng.integers(1, 3))
        if t == "T03" and m in ("2026-07", "2026-08", "2026-09"):   # vacancy, then new rep ramping
            field_days = {"2026-07": 9, "2026-08": 6, "2026-09": 12}[m]
        days_rows.append((t, m, avail, field_days))
        capacity = field_days / (avail - 1.5)  # 1.0 = a normal month

        for p, cfg in PRODUCT_MODEL.items():
            base_market = (cfg["market"][0] + cfg["market"][1]) / 2 * (0.7 + 0.6 * (int(t[1:]) % 7) / 6)
            market = base_market * (1 + cfg["market_growth"]) ** i * rng.normal(1, 0.02)
            share = cfg["share"] + cfg["share_drift"] * i
            planned_sales = market * share  # the plan the target was built from

            if p == "Cardiora" and t == "T07" and m >= "2026-06":   # competitor launch
                months_since = MONTHS.index(m) - MONTHS.index("2026-06") + 1
                market *= 1 + 0.08 * months_since     # market grows fast...
                share -= 0.010 * months_since         # ...but we lose share
            if p == "Cardiora" and t == "T13":                         # strong performer
                share *= 1.12
            if p == "Cardiora" and t == "T16" and i >= 6:              # neglected P1
                share *= 0.92
            if p == "Cardiora" and t == "T03":                         # vacancy hits sales
                share *= {"2026-07": 0.93, "2026-08": 0.86, "2026-09": 0.84}.get(m, 1.0)

            sales = market * share * rng.normal(1, 0.015)
            sales_rows.append((t, p, m, round(sales), round(planned_sales), round(market)))

            tgt = targets[targets["product"] == p].set_index("segment")
            for seg, n in universe.items():
                reach = tgt.loc[seg, "reach_target"] * rng.normal(1.0, 0.06)
                freq = tgt.loc[seg, "frequency_target"] * rng.normal(1.0, 0.08)
                if t == "T03":                     # vacancy reduces activity on every brand
                    reach *= min(1.0, capacity * 1.05)
                    freq *= min(1.0, 0.6 + 0.4 * capacity)
                if t == "T13" and p == "Cardiora":
                    reach, freq = reach * 1.15, freq * 1.12
                if t == "T16" and p == "Cardiora" and seg == "A" and i >= 6:
                    reach *= 0.72
                if t == "T16" and p == "Dermalyn" and seg in ("A", "B") and i >= 6:
                    reach, freq = reach * 2.0, freq * 3.2
                reached = int(min(n, max(0, round(n * reach))))
                calls = int(round(reached * max(freq, 0)))
                activity_rows.append((t, p, m, seg, n, reached, calls))

fact_sales = pd.DataFrame(sales_rows, columns=["territory_id", "product", "month", "sales_value", "target_value", "market_value"])
fact_activity = pd.DataFrame(activity_rows, columns=["territory_id", "product", "month", "segment", "customers_in_segment", "customers_reached", "calls"])
fact_field_days = pd.DataFrame(days_rows, columns=["territory_id", "month", "available_days", "field_days"])

metric_definitions = pd.DataFrame([
    ("sales_value", "Sales", "Value of sales in the territory for the month (AUD).",
     "SUM(sales_value)", "Higher is better", "Read alongside target and market share, not on its own."),
    ("target_value", "Target", "The sales target set for the territory for the month (AUD).",
     "SUM(target_value)", "Not applicable", "Targets are set from the brand plan at the start of the year."),
    ("sales_vs_target_pct", "Sales vs target", "Sales as a percentage of target.",
     "SUM(sales_value) / SUM(target_value) x 100", "Higher is better",
     "On track at 98% or above, watch from 90% to 98%, off track below 90%."),
    ("sales_gap_value", "Gap to target", "Sales minus target for the month (AUD). Negative means behind target.",
     "SUM(sales_value) - SUM(target_value)", "Higher is better", "Shows the size of the shortfall or over-delivery in dollars."),
    ("market_share_pct", "Market share", "Our sales as a share of total market sales for the brand's market definition.",
     "SUM(sales_value) / SUM(market_value) x 100", "Higher is better",
     "Shows whether we are winning against competitors, even when the market itself is growing or shrinking."),
    ("market_share_change_pts", "Market share change", "Change in market share versus the previous three months combined, in percentage points.",
     "Market share (month) - market share (previous 3 months combined)", "Higher is better",
     "A fall while sales are still near target often means the market is growing faster than we are."),
    ("reach_pct", "Reach", "Share of target customers in a segment seen at least once in the month.",
     "SUM(customers_reached) / SUM(customers_in_segment) x 100", "Higher is better, up to the target",
     "Compare with the reach target for the product's priority. A P1 brand needs far higher A-segment reach than a P3 brand."),
    ("reach_gap_pts", "Reach gap", "Reach minus the reach target, in percentage points. Negative means below target.",
     "Reach - reach target", "Higher is better", "Shows how far coverage of a segment is from what the brand's priority requires."),
    ("frequency", "Frequency", "Average calls per customer reached in the month.",
     "SUM(calls) / SUM(customers_reached)", "Close to target",
     "Under 80% of target is under-servicing; over 150% of target is over-servicing that takes time from priority brands."),
    ("call_rate", "Call rate", "Average calls per field day, across all products.",
     "SUM(calls) / SUM(field_days)", "Higher is better, within reason",
     "If field days are normal, a normal call rate with poor priority-brand coverage points to effort spent on the wrong customers or brands. If field days are low, a normal call rate is expected: the issue is time in the field, not where the effort goes."),
    ("field_day_pct", "Field days", "Field days worked as a percentage of available days.",
     "SUM(field_days) / SUM(available_days) x 100", "Higher is better",
     "A sharp drop usually means leave, vacancy or training, and it flows through to reach and sales."),
], columns=["metric_id", "name", "definition", "calculation", "good_direction", "how_to_read"])

for name, df in {
    "dim_territory": territories, "dim_product": products, "targets": targets,
    "fact_sales": fact_sales, "fact_activity": fact_activity,
    "fact_field_days": fact_field_days, "metric_definitions": metric_definitions,
}.items():
    df.to_csv(OUT / f"{name}.csv", index=False)

(OUT / "metadata.json").write_text(json.dumps({
    "refreshed_at": "12 Oct 2026, 6:00 am AEDT",
    "latest_period": MONTHS[-1],
    "source_note": "Synthetic data. No real company, product, customer or sales data.",
}, indent=2))
print(f"Wrote {len(fact_sales)} sales rows and {len(fact_activity)} activity rows to {OUT}")
