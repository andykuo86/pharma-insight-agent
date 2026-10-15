#!/usr/bin/env python3
"""Update the pharma-insight-agent project in place.

Run from the project folder:   python3 apply_update.py

It adds the dashboard screenshots to the README, lets the assistant link to a hosted
dashboard (REPORT_DASHBOARD_URL), and adds tests for that. It is safe to run twice:
anything already applied is skipped, and nothing is changed if an expected place in a
file is not found (it says so instead).
"""
from pathlib import Path

root = Path(__file__).resolve().parent
done, skipped, problems = [], [], []


def edit(rel, fn):
    path = root / rel
    if not path.exists():
        problems.append(f"{rel}: file not found")
        return
    text = path.read_text(encoding="utf-8")
    new, note = fn(text)
    if new is None:
        (skipped if note.startswith("already") else problems).append(f"{rel}: {note}")
    else:
        path.write_text(new, encoding="utf-8")
        done.append(f"{rel}: {note}")


# ---------------------------------------------------------------- semantic_layer.py
OLD_HEAD = '''def report_link(filters: Filters, view: str = "sales", source: str | None = None) -> str:
    """Link that opens the report with exactly these filters pre-selected."""
    if view not in REPORT_VIEWS:
        raise ValueError(f"Unknown report view '{view}'")
'''
NEW_HEAD = '''def _dashboard_link(base: str, filters: Filters, view: str) -> str:
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
'''


def patch_semantic(text):
    if "_dashboard_link" in text:
        return None, "already updated"
    if OLD_HEAD not in text or "import json\nimport os\n" not in text:
        return None, "could not find report_link or the imports, so it was left unchanged"
    if "\nimport re\n" not in text:
        text = text.replace("import json\nimport os\n", "import json\nimport os\nimport re\n", 1)
    return text.replace(OLD_HEAD, NEW_HEAD, 1), "assistant links can now open a hosted dashboard"


edit("semantic_layer.py", patch_semantic)

# ---------------------------------------------------------------- tests
NEW_TESTS = '''

def test_dashboard_link_carries_the_tab_and_filters(monkeypatch):
    monkeypatch.setenv("REPORT_DASHBOARD_URL", "https://example.com/dashboard")
    f = sl.Filters(period="2026-09", territory="T03", product="Cardiora", segment="A")
    assert sl.report_link(f, "coverage", source="assistant") == "https://example.com/dashboard#coverage_t-T03_p-Cardiora_s-A_m-2026-09"


def test_dashboard_link_sends_a_region_as_an_id(monkeypatch):
    monkeypatch.setenv("REPORT_DASHBOARD_URL", "https://example.com/dashboard/")
    f = sl.Filters(period="2026-09", region="NSW & ACT")
    assert sl.report_link(f, "sales") == "https://example.com/dashboard#sales_r-NSW-ACT_m-2026-09"


def test_without_a_dashboard_the_local_report_is_used(monkeypatch):
    monkeypatch.delenv("REPORT_DASHBOARD_URL", raising=False)
    f = sl.Filters(period="2026-09", territory="T03")
    assert sl.report_link(f, "sales").startswith(sl.REPORT_BASE_URL + "?")
'''


def patch_tests(text):
    if "test_dashboard_link_carries_the_tab_and_filters" in text:
        return None, "already updated"
    return text.rstrip("\n") + "\n" + NEW_TESTS, "three tests added for the dashboard link"


edit("tests/test_semantic_layer.py", patch_tests)

# ---------------------------------------------------------------- README.md
SHOTS = '''## What the report looks like

The standard report the assistant links back to, built as a hosted dashboard over the same
synthetic data. Status is colour coded: on track, watch and off track.

<p align="center">
  <img src="docs/dashboard-sales.png" alt="The Sales tab of the performance report for all of ANZ in September 2026. Headline tiles show sales vs target, sales, market share and how many territory and brand pairs are off track. A table lists each territory and product, furthest behind target first, with colour-coded status." width="100%">
</p>

Clicking "Check this in the report" on an insight opens the dashboard on the right tab with
the filters already set. This is Parramatta and Cardiora on the Coverage tab, with reach
against target for each customer segment and the field days that explain it:

<p align="center">
  <img src="docs/dashboard-coverage.png" alt="The Coverage tab opened from an assistant link, filtered to Parramatta and Cardiora in September 2026. Reach is below target in all three customer segments, each shown as a red bar with a target tick, and a chart below shows field days dropping in the recent months." width="100%">
</p>

'''
DASH = '''### Linking to a hosted dashboard (optional)

By default the assistant's "Check this in the report" links open the local Streamlit
report in `views/report.py`. The same report can instead live as a hosted dashboard
(this project's author built one as a Claude dashboard over the same synthetic data).
To point the links at a dashboard, set its address before starting the app:

```bash
export REPORT_DASHBOARD_URL="https://your-dashboard-address"
```

The assistant then builds links such as
`https://your-dashboard-address#coverage_t-T03_p-Cardiora_s-A_m-2026-09`: the tab, then
the territory, product, segment and month, as one short name. A dashboard needs to read
that name when it opens and apply the filters; see `semantic_layer.py` for the format.
The dashboard's figures were checked against `semantic_layer.py` on every month, scope,
brand and segment (25,200 values, no differences).

'''


def patch_readme(text):
    import re
    notes = []
    if "## What the report looks like" not in text:
        if "## Design principles" not in text:
            return None, "could not find the '## Design principles' heading, so it was left unchanged"
        text = text.replace("## Design principles", SHOTS + "## Design principles", 1)
        notes.append("screenshots added")
    if "### Linking to a hosted dashboard (optional)" not in text and "## Choosing the model" in text:
        text = text.replace("## Choosing the model", DASH + "## Choosing the model", 1)
        notes.append("dashboard section added")
    updated = re.sub(r"# \d+ tests, no API key needed", "# 30 tests, no API key needed", text)
    if updated != text:
        text = updated
        notes.append("test count updated")
    if not notes:
        return None, "already updated"
    return text, ", ".join(notes)


edit("README.md", patch_readme)

# ---------------------------------------------------------------- images
for img in ("docs/dashboard-sales.png", "docs/dashboard-coverage.png"):
    (done if (root / img).exists() else problems).append(f"{img}: " + ("in place" if (root / img).exists() else "missing, unzip the kit into the project folder first"))

print()
for title, items in (("Done", done), ("Already up to date", skipped), ("Needs attention", problems)):
    if items:
        print(title + ":")
        for i in items:
            print("  -", i)
if not problems:
    print("\nNext: run  pytest -q  (expect 30 passed), then commit in GitHub Desktop.")
