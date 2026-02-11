import os
import tempfile
import time
from typing import Any, Dict

import streamlit as st
import pandas as pd

from src.observability.dashboard.services.app_context import get_ingestion_pipeline, get_document_manager

def render_ingestion_manager_page() -> None:
    st.title("Ingestion Manager 📥")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Ingestion Settings")
    collection_name = st.sidebar.text_input("Target Collection", value="knowledge_hub")
    force_ingest = st.sidebar.checkbox("Force Ingest (Ignore Hash)", value=False)

    # --- Main Area: File Upload ---
    st.subheader("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["pdf"], 
        help="Currently supports PDF files only."
    )

    if uploaded_file:
        st.info(f"File selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        if st.button("Start Ingestion", type="primary"):
            _handle_ingestion(uploaded_file, collection_name, force_ingest)
            
    st.divider()
    
    # --- Management Section ---
    _render_management_section(collection_name)

def _render_management_section(collection_name: str) -> None:
    """Render the document management section (Deletion)."""
    st.subheader("Manage Documents")
    
    try:
        doc_manager = get_document_manager()
        docs = doc_manager.list_documents(collection=collection_name)
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return

    if not docs:
        st.info(f"No documents found in collection '{collection_name}'.")
        return

    # Create a simple table for display
    df_data = [
        {
            "Source Path": d.source_path,
            "Chunks": d.chunk_count,
            "Images": d.image_count
        }
        for d in docs
    ]
    df = pd.DataFrame(df_data)
    
    st.markdown(f"**Total Documents:** {len(docs)}")
    
    # Use dataframe with selection for deletion
    selection = st.dataframe(
        df,
        width='stretch',
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True
    )
    
    selected_rows = selection.get("selection", {}).get("rows", [])
    if selected_rows:
        selected_idx = selected_rows[0]
        doc_to_delete = docs[selected_idx]
        
        st.warning(f"Selected: `{doc_to_delete.source_path}`")
        if st.button("🗑️ Delete Selected Document", type="primary"):
            with st.spinner("Deleting..."):
                res = doc_manager.delete_document(doc_to_delete.source_path, collection_name)
                if res.success:
                    st.success(f"Deleted successfully! (Chunks: {res.deleted_chunks}, Images: {res.deleted_images})")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Deletion failed: {res.message}")

def _handle_ingestion(uploaded_file: Any, collection_name: str, force: bool) -> None:
    """Handle the ingestion process with UI feedback."""
    
    # 1. Initialize Pipeline
    try:
        pipeline = get_ingestion_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return

    # 2. Save uploaded file to temp
    # We use a suffix to help loader identify file type
    suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # 3. Run Ingestion with Progress
        progress_bar = st.progress(0, text="Starting ingestion...")
        status_container = st.status("Ingestion in progress...", expanded=True)
        
        # Progress Callback
        def on_progress(stage: str, data: Dict[str, Any]) -> None:
            _update_progress(stage, data, progress_bar, status_container)

        # Execute
        result = pipeline.ingest(
            collection=collection_name,
            file_path=tmp_path,
            force=force,
            on_progress=on_progress
        )
        
        status_container.update(label="Ingestion Completed!", state="complete", expanded=False)
        progress_bar.progress(100, text="Done!")
        
        # 4. Show Result
        if result.skipped:
            st.warning(f"Ingestion skipped: File hash {result.file_hash[:8]} already exists. (Use 'Force Ingest' to override)")
        else:
            st.success(f"Successfully ingested `{uploaded_file.name}` into `{collection_name}`!")
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks", len(result.chunks))
            
            upsert_count = len(result.upsert.records) if result.upsert else 0
            col2.metric("Vectors Upserted", upsert_count)
            
            images_count = 0
            if result.document and result.document.metadata.get("images"):
                images_list = result.document.metadata.get("images")
                if isinstance(images_list, list):
                    images_count = len(images_list)
            col3.metric("Images Extracted", images_count)

    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # 5. Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def _update_progress(stage: str, data: Dict[str, Any], progress_bar: Any, status_container: Any) -> None:
    """Map pipeline stages to UI progress updates."""
    
    mapping = {
        "start": (0, "Initializing..."),
        "integrity": (5, "Checking file integrity..."),
        "hash_calculated": (10, "Hash calculated."),
        "skipped": (100, "Skipped."),
        "loaded": (20, "Document loaded."),
        "split": (30, "Document split into chunks."),
        "transformed": (50, "Enriching & Transforming..."),
        "encoded": (70, "Generating Embeddings..."),
        "upserted": (85, "Upserting vectors..."),
        "bm25_built": (90, "Updating BM25 index..."),
        "image_store": (95, "Storing images..."),
        "finalize": (99, "Finalizing..."),
        "complete": (100, "Complete.")
    }
    
    if stage in mapping:
        pct, text = mapping[stage]
        progress_bar.progress(pct, text=text)
        status_container.write(f"✅ {text}")
        
        # Add detailed info for specific stages
        if stage == "split":
            chunks = data.get("chunks", [])
            status_container.write(f"   - Generated {len(chunks)} chunks")
        elif stage == "encoded":
            batch = data.get("batch")
            if batch:
                dense = len(batch.dense_vectors)
                status_container.write(f"   - Encoded {dense} vectors")
