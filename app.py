"""Entry point. Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Commercial insight demo", layout="wide")

# The assistant is the front door, served at the root URL. The report needs its own path
# (/report) so that links from the assistant can open it with filters pre-selected.
# Streamlit always serves the default page at the root and ignores its url_path.
page = st.navigation([
    st.Page("views/assistant.py", title="Insight assistant", default=True),
    st.Page("views/report.py", title="Performance report", url_path="report"),
])
page.run()
