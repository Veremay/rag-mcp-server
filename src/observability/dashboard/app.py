import streamlit as st

from src.observability.dashboard.pages.overview import render_overview
from src.observability.dashboard.pages.data_browser import render_data_browser_page
from src.observability.dashboard.pages.ingestion_manager import render_ingestion_manager_page
from src.observability.dashboard.pages.ingestion_traces import render_ingestion_traces_page


def placeholder_page() -> None:
    st.title("Coming Soon")
    st.info("This page is under construction.")


# Define pages
overview_page = st.Page(render_overview, title="System Overview", icon="🏠")
ingestion_traces_page = st.Page(
    render_ingestion_traces_page,
    title="Ingestion Traces",
    icon="⏱️",
    url_path="ingestion-traces",
)
trace_list_page = st.Page(
    placeholder_page, title="Trace List", icon="📋", url_path="traces"
)
trace_detail_page = st.Page(
    placeholder_page, title="Trace Detail", icon="🔍", url_path="trace-detail"
)
knowledge_page = st.Page(
    render_data_browser_page, title="Data Browser", icon="📚", url_path="data-browser"
)
ingestion_page = st.Page(
    render_ingestion_manager_page,
    title="Ingestion Manager",
    icon="📥",
    url_path="ingestion",
)
retrieval_page = st.Page(
    placeholder_page, title="Retrieval Lab", icon="🧪", url_path="retrieval"
)
eval_page = st.Page(placeholder_page, title="Evaluation", icon="📊", url_path="evaluation")

# Navigation
pg = st.navigation(
    {
        "General": [overview_page],
        "Observability": [ingestion_traces_page, trace_list_page, trace_detail_page],
        "Knowledge & Tools": [knowledge_page, ingestion_page, retrieval_page, eval_page],
    }
)

st.set_page_config(page_title="RAG Dashboard", layout="wide")
pg.run()
