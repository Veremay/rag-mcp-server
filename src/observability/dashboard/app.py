import sys
from pathlib import Path

# Streamlit 在子进程运行本文件时，项目根通常不在 PYTHONPATH，导致 No module named 'src'
# 在首次 import src 前把项目根加入 path，保证无论用 streamlit run 还是 python scripts/start_dashboard.py 都能正常导入
_project_root = Path(__file__).resolve().parents[3]
if _project_root.exists() and str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.observability.dashboard.pages.overview import render_overview
from src.observability.dashboard.pages.data_browser import render_data_browser_page
from src.observability.dashboard.pages.ingestion_manager import render_ingestion_manager_page
from src.observability.dashboard.pages.ingestion_traces import render_ingestion_traces_page
from src.observability.dashboard.pages.query_traces import render_query_traces_page
from src.observability.dashboard.pages.evaluation_panel import render_evaluation_panel


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
query_traces_page = st.Page(
    render_query_traces_page,
    title="Query Traces",
    icon="🔍",
    url_path="query-traces",
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
eval_page = st.Page(render_evaluation_panel, title="Evaluation", icon="📊", url_path="evaluation")

# Navigation
pg = st.navigation(
    {
        "General": [overview_page],
        "Observability": [ingestion_traces_page, query_traces_page],
        "Knowledge & Tools": [knowledge_page, ingestion_page, retrieval_page, eval_page],
    }
)

st.set_page_config(page_title="RAG Dashboard", layout="wide")
pg.run()
