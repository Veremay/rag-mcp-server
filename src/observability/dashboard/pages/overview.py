import streamlit as st
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.dashboard.services.config_service import ConfigService


def render_overview() -> None:
    st.header("System Overview")

    config_service = ConfigService()
    info = config_service.get_component_info()
    settings = config_service.get_settings()

    # Component Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("LLM")
        st.write(f"Provider: **{info['llm']['provider']}**")
        st.write(f"Model: **{info['llm']['model']}**")

    with col2:
        st.subheader("Embedding")
        st.write(f"Provider: **{info['embedding']['provider']}**")
        st.write(f"Model: **{info['embedding']['model']}**")

    with col3:
        st.subheader("Retrieval")
        st.write(f"Sparse: **{info['retrieval']['sparse']}**")
        st.write(f"Fusion: **{info['retrieval']['fusion']}**")
        st.write(f"Top-K: **{info['retrieval']['top_k']}**")

    st.divider()

    # Vector Store Stats
    st.subheader("Knowledge Base Stats")
    try:
        store = VectorStoreFactory.create(settings)
        stats = store.get_collection_stats()

        m1, m2, m3 = st.columns(3)
        m1.metric("Documents/Chunks", stats.get("count", "N/A"))
        m2.metric("Backend", stats.get("backend", "N/A"))
        m3.metric("Collection", stats.get("collection_name", "N/A"))

        st.caption(f"Persist Path: {stats.get('persist_path', 'N/A')}")

    except Exception as e:
        st.error(f"Failed to load vector store stats: {e}")
