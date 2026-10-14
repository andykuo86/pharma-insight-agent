"""Mock 'standard report' - stands in for the Power BI or Tableau report.

The filters come from the URL, so a link such as
    /report?period=2026-09&territory=T03&product=Cardiora&view=coverage
opens the report already filtered. Every number on this page comes from
semantic_layer.py, the same source the insight assistant uses.
"""
import pandas as pd
import streamlit as st

import semantic_layer as sl

ALL = "All"
VIEW_LABELS = {"sales": "Sales", "coverage": "Customer coverage", "field": "Field capacity", "definitions": "Metric definitions"}
STATUS_COLOURS = {
    "On track": "#2F6F3E", "On target": "#2F6F3E",
    "Watch": "#A86B00",
    "Off track": "#B4441C", "Under-servicing": "#B4441C", "Over-servicing": "#A86B00",
}

# --------------------------------------------------------------- filters from URL
params = st.query_params.to_dict()
url_filters = sl.Filters.from_params(params)
view = params.get("view") if params.get("view") in VIEW_LABELS else "sales"
came_from_assistant = params.get("from") == "assistant"

terr = sl.territories()


def pick(label: str, options: list[str], current: str | None, fmt=None) -> str | None:
    """Sidebar select box pre-set from the URL. Returns None for 'All'."""
    choices = [ALL] + options
    index = choices.index(current) if current in choices else 0
    value = st.sidebar.selectbox(label, choices, index=index, format_func=fmt or (lambda x: x))
    return None if value == ALL else value


st.sidebar.header("Filters")
period = st.sidebar.selectbox(
    "Month", sl.periods()[::-1], index=sl.periods()[::-1].index(url_filters.period),
    format_func=lambda p: pd.Period(p).strftime("%B %Y"),
)
region = pick("Region", sorted(terr["region"].unique()), url_filters.region)
terr_scope = terr if region is None else terr[terr["region"] == region]
names = dict(zip(terr["territory_id"], terr["territory_name"]))
territory = pick("Territory", terr_scope["territory_id"].tolist(), url_filters.territory,
                 fmt=lambda t: t if t == ALL else f"{names[t]} ({t})")
product = pick("Product", sl.products()["product"].tolist(), url_filters.product)
segment = pick("Customer segment", ["A", "B", "C"], url_filters.segment)

filters = sl.Filters(period=period, region=region, territory=territory, product=product, segment=segment)

view = st.radio("View", list(VIEW_LABELS), index=list(VIEW_LABELS).index(view),
                format_func=VIEW_LABELS.get, horizontal=True, label_visibility="collapsed")

# Keep the address bar in step with the filters, so any view can be shared as a link.
st.query_params.from_dict(filters.to_params() | {"view": view})

# ------------------------------------------------------------------------ header
st.title("Commercial performance report")
st.subheader(sl.describe(filters))
meta = sl.metadata()
st.caption(f"Data refreshed {meta['refreshed_at']}. {meta['source_note']}")

if came_from_assistant:
    st.info("You opened this view from the insight assistant. These are the same numbers the assistant "
            "used, filtered the same way. Change the filters on the left to explore further.")


def status_style(value):
    colour = STATUS_COLOURS.get(value)
    return f"color: {colour}; font-weight: 600" if colour else ""


def show_table(df: pd.DataFrame, columns: dict[str, str], status_cols: list[str]):
    table = df[list(columns)].rename(columns=columns)
    styled = table.style.map(status_style, subset=[columns[c] for c in status_cols])
    st.dataframe(styled, hide_index=True, width="stretch", column_config={
        "Sales (AUD)": st.column_config.NumberColumn(format="$%d"),
        "Target (AUD)": st.column_config.NumberColumn(format="$%d"),
        "Sales vs target": st.column_config.NumberColumn(format="%.1f%%"),
        "Market share": st.column_config.NumberColumn(format="%.1f%%"),
        "Share change (pts)": st.column_config.NumberColumn(format="%+.1f"),
        "Reach": st.column_config.NumberColumn(format="%.1f%%"),
        "Reach target": st.column_config.NumberColumn(format="%.1f%%"),
        "Field days": st.column_config.NumberColumn(format="%.1f%%"),
    })


# ------------------------------------------------------------------------- views
if view == "sales":
    total = sl.sales_performance(filters, by=()).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales vs target", f"{total.sales_vs_target_pct:.1f}%", help=sl.definition("sales_vs_target_pct")["how_to_read"])
    c2.metric("Sales", f"${total.sales_value:,.0f}", f"Target ${total.target_value:,.0f}", delta_color="off")
    c3.metric("Market share", f"{total.market_share_pct:.1f}%", f"{total.market_share_change_pts:+.1f} pts vs prior 3 months",
              help=sl.definition("market_share_change_pts")["how_to_read"])
    c4.metric("Status", total.status)

    st.markdown("#### Sales by territory and product")
    show_table(sl.sales_performance(filters), {
        "territory_name": "Territory", "product": "Product", "priority": "Priority",
        "sales_value": "Sales (AUD)", "target_value": "Target (AUD)", "sales_vs_target_pct": "Sales vs target",
        "market_share_pct": "Market share", "market_share_change_pts": "Share change (pts)", "status": "Status",
    }, ["status"])

    st.markdown("#### Last 12 months")
    tr = sl.trend(filters).set_index("month")
    left, right = st.columns(2)
    left.caption("Sales vs target (%)")
    left.line_chart(tr[["sales_vs_target_pct"]], height=220)
    right.caption("Market share (%)")
    right.line_chart(tr[["market_share_pct"]], height=220)

elif view == "coverage":
    st.markdown("#### Reach and frequency against target")
    st.caption("Reach targets depend on the product's priority: a P1 brand needs far higher A-segment reach than a P3 brand.")
    show_table(sl.coverage(filters), {
        "territory_name": "Territory", "product": "Product", "priority": "Priority", "segment": "Segment",
        "reach_pct": "Reach", "reach_target_pct": "Reach target", "reach_status": "Reach status",
        "frequency": "Frequency", "frequency_target": "Frequency target", "frequency_status": "Frequency status",
    }, ["reach_status", "frequency_status"])

elif view == "field":
    st.markdown("#### Field days and call rate")
    st.caption("Across all products. Product and segment filters do not apply to this view.")
    show_table(sl.field_capacity(filters), {
        "territory_name": "Territory", "manager": "Manager", "available_days": "Available days",
        "field_days": "Days in field", "field_day_pct": "Field days", "calls": "Calls", "call_rate": "Call rate",
    }, [])

else:
    st.markdown("#### Metric definitions")
    st.caption("Governed definitions. The report and the insight assistant both use these.")
    st.dataframe(sl.definitions().drop(columns="metric_id").rename(columns={
        "name": "Metric", "definition": "What it means", "calculation": "How it is calculated",
        "good_direction": "Good direction", "how_to_read": "How to read it",
    }), hide_index=True, width="stretch")

with st.expander("Link to this view"):
    st.code(sl.report_link(filters, view), language=None)
    st.caption("Anyone with this link sees the report with the same filters.")
