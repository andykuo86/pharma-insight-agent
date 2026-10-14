"""Insight assistant page.

Writes each manager's monthly brief, links every insight to the matching report view,
and answers follow-up questions. Every answer is checked against the data the tools
returned before it is shown.
"""
import json

import pandas as pd
import streamlit as st

import semantic_layer as sl
from assistant.agent import InsightAgent
from assistant.llm import api_key_configured, configured_provider, get_client, model_name
from assistant.validation import evidence_links, validate

st.title("Insight assistant")
st.caption("Your monthly brief, with every insight linked to the performance report so you can check it yourself.")

if not api_key_configured():
    st.warning("Set an API key to use the assistant.")
    st.markdown(
        "Google Gemini has a free tier: create a key in Google AI Studio, then in Terminal:\n\n"
        "```bash\nexport GEMINI_API_KEY=\"your-key-here\"\nstreamlit run app.py\n```\n\n"
        "Claude (`ANTHROPIC_API_KEY`), Groq (`GROQ_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`) also work. "
        "The performance report works without a key."
    )
    st.stop()

# ---------------------------------------------------------------- who is asking
terr = sl.territories()
managers = terr.drop_duplicates("manager")[["manager", "region"]].sort_values("manager")
region_of = dict(zip(managers.manager, managers.region))

st.sidebar.caption(f"Model: {configured_provider()} / {model_name(configured_provider())}")
st.sidebar.header("Your view")
manager = st.sidebar.selectbox("Manager", managers.manager, format_func=lambda m: f"{m} ({region_of[m]})")
months = sl.periods()[::-1]
period = st.sidebar.selectbox("Month", months, format_func=lambda p: pd.Period(p).strftime("%B %Y"))
region = region_of[manager]
context_key = f"{manager}|{period}"

if st.session_state.get("context_key") != context_key:      # new manager or month: start fresh
    st.session_state.update(context_key=context_key, brief=None, brief_result=None,
                            chat_api=[], chat_view=[], evidence=[])


@st.cache_resource
def get_agent() -> InsightAgent:
    return InsightAgent(get_client())


agent = get_agent()


def md(text: str) -> str:
    """Escape dollar signs so Streamlit shows them instead of treating $...$ as a maths formula."""
    return str(text).replace("$", "\\$")


def show_check(validation):
    if validation.ok:
        st.success(validation.summary(), icon="✅")
    else:
        st.warning(validation.summary(), icon="⚠️")


def show_evidence(tool_calls):
    with st.expander("Data the assistant looked at"):
        for call in tool_calls:
            st.markdown(f"- `{call['tool']}` {json.dumps(call['input']) if call['input'] else ''}")


# ------------------------------------------------------------------------ brief
if st.session_state.brief is None:
    st.markdown(f"#### {manager}, {region}, {pd.Period(period).strftime('%B %Y')}")
    if st.button("Write my monthly brief", type="primary"):
        with st.spinner("Reading the report data and writing your brief..."):
            try:
                brief, result = agent.monthly_brief(manager, region, period)
            except Exception as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    st.error("The model's usage limit has been reached. Free tiers have small daily limits. "
                             "Try again later, switch model with LLM_MODEL, or use another provider's key.")
                else:
                    st.error(f"The brief could not be written: {exc}")
                st.stop()
        st.session_state.update(brief=brief, brief_result=result, evidence=list(result.evidence))
        st.session_state.chat_api = [
            {"role": "user", "content": f"Context: I am {manager}, manager of the {region} region, looking at "
                                        f"{pd.Period(period).strftime('%B %Y')} ({period}). This is the brief you wrote "
                                        f"for me:\n{json.dumps(brief)}"},
            {"role": "assistant", "content": "Thanks. What would you like to look at more closely?"},
        ]
        st.rerun()
    st.stop()

brief, result = st.session_state.brief, st.session_state.brief_result
links_available = evidence_links(st.session_state.evidence)

st.markdown(f"#### {md(brief.get('headline', ''))}")
show_check(result.validation)

for insight in brief["insights"]:
    with st.container(border=True):
        st.markdown(f"**{md(insight['title'])}**")
        st.markdown(f"**What is happening.** {md(insight['what_is_happening'])}")
        st.markdown(f"**Why it matters.** {md(insight['why_it_matters'])}")
        st.markdown(f"**Suggested action.** {md(insight['suggested_action'])}")
        st.markdown(f"**Coaching question.** _{md(insight['coaching_question'])}_")
        try:
            d = sl.definition(insight.get("metric_id", ""))
            st.caption(f"How to read {d['name'].lower()}: {d['definition']} {d['how_to_read']}")
        except KeyError:
            pass
        link = insight.get("report_link", "")
        if link in links_available:
            st.link_button("Check this in the report", link)
        else:
            st.caption("No verified report link for this insight.")

show_evidence(result.tool_calls)

# -------------------------------------------------------------- follow-up chat
st.divider()
st.markdown("#### Ask a follow-up question")

for turn in st.session_state.chat_view:
    with st.chat_message(turn["role"]):
        st.markdown(md(turn["text"]) if turn["role"] == "assistant" else turn["text"])
        if turn["role"] == "assistant":
            show_check(turn["validation"])

question = st.chat_input("For example: why is Parramatta behind on Cardiora?")
if question:
    st.session_state.chat_view.append({"role": "user", "text": question})
    st.session_state.chat_api.append({"role": "user", "content": question})
    with st.spinner("Checking the report data..."):
        try:
            answer = agent.run(st.session_state.chat_api)
        except Exception as exc:
            st.error(f"The assistant could not answer: {exc}")
            st.stop()
    st.session_state.evidence.extend(answer.evidence)
    checked = validate(answer.text, st.session_state.evidence)   # check against everything seen so far
    st.session_state.chat_view.append({"role": "assistant", "text": answer.text, "validation": checked})
    st.rerun()
