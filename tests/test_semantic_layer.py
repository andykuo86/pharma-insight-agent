"""Tests for the semantic layer and the report.

Run:  pytest
"""
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from streamlit.testing.v1 import AppTest

import semantic_layer as sl

SEP = "2026-09"


def test_ratios_are_sum_over_sum_not_average_of_ratios():
    f = sl.Filters(period=SEP, region="NSW & ACT", product="Cardiora")
    detail = sl.sales_performance(f)
    total = sl.sales_performance(f, by=()).iloc[0]
    expected = round(100 * detail.sales_value.sum() / detail.target_value.sum(), 1)
    assert total.sales_vs_target_pct == expected


def test_report_link_round_trips_to_the_same_filters():
    f = sl.Filters(period=SEP, territory="T03", product="Cardiora", segment="A")
    params = {k: v[0] for k, v in parse_qs(urlparse(sl.report_link(f, "coverage")).query).items()}
    assert sl.Filters.from_params(params) == f
    assert params["view"] == "coverage"


def test_invalid_url_values_are_ignored():
    f = sl.Filters.from_params({"territory": "T99", "product": "Unknown", "period": "1999-01"})
    assert f == sl.Filters(period=sl.latest_period())


@pytest.mark.parametrize("territory, expected_status", [("T03", "Off track"), ("T13", "On track")])
def test_built_in_scenarios_are_visible(territory, expected_status):
    row = sl.sales_performance(sl.Filters(period=SEP, territory=territory, product="Cardiora")).iloc[0]
    assert row.status == expected_status


def test_report_shows_the_same_numbers_as_the_semantic_layer():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "views" / "report.py"), default_timeout=30)
    at.query_params.update({"period": SEP, "territory": "T07", "product": "Cardiora"})
    at.run()
    assert not at.exception
    shown = {m.label: m.value for m in at.metric}
    total = sl.sales_performance(sl.Filters(period=SEP, territory="T07", product="Cardiora"), by=()).iloc[0]
    assert shown["Sales vs target"] == f"{total.sales_vs_target_pct:.1f}%"
    assert shown["Market share"] == f"{total.market_share_pct:.1f}%"
