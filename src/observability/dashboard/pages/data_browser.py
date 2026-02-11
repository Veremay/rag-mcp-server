import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from src.observability.dashboard.services.app_context import get_document_manager
from src.ingestion.document_manager import DocumentManager, DocumentInfo, DocumentDetail

def render_data_browser_page() -> None:
    st.title("Data Browser 📂")
    
    try:
        doc_manager = get_document_manager()
    except Exception as e:
        st.error(f"Failed to initialize DocumentManager: {e}")
        return

    # Sidebar
    st.sidebar.header("Filter & Actions")
    
    # Get stats to know current collection
    try:
        stats = doc_manager.get_collection_stats()
        current_collection = stats.collection_name
    except Exception:
        current_collection = "knowledge_hub" # Default fallback
        
    collection = st.sidebar.text_input("Active Collection", value=current_collection, disabled=True)
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Main Area: List Documents
    try:
        docs = doc_manager.list_documents(collection=collection)
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return

    if not docs:
        st.info("No documents found in the current collection.")
        return

    # Summary Metrics
    total_docs = len(docs)
    total_chunks = sum(d.chunk_count for d in docs)
    total_images = sum(d.image_count for d in docs)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", total_docs)
    m2.metric("Total Chunks", total_chunks)
    m3.metric("Total Images", total_images)
    
    st.divider()
    
    # Document Table
    st.subheader("Document List")
    
    df_data = [
        {
            "Source Path": d.source_path,
            "Chunks": d.chunk_count,
            "Images": d.image_count,
        }
        for d in docs
    ]
    df = pd.DataFrame(df_data)
    
    # Use selection to drive detail view
    selection = st.dataframe(
        df, 
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True
    )
    
    selected_rows = selection.get("selection", {}).get("rows", [])
    
    if selected_rows:
        selected_index = selected_rows[0]
        selected_doc_info = docs[selected_index]
        source_path = selected_doc_info.source_path
        
        st.divider()
        st.subheader(f"📄 Document Details")
        st.markdown(f"**Source:** `{source_path}`")
        
        col_act1, col_act2 = st.columns([1, 5])
        with col_act1:
            if st.button("🗑️ Delete Document", type="primary"):
                with st.spinner("Deleting document and associated resources..."):
                    res = doc_manager.delete_document(source_path, collection)
                    if res.success:
                        st.success(f"Deleted successfully! (Chunks: {res.deleted_chunks}, Images: {res.deleted_images})")
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Deletion failed: {res.message}")
        
        # Detail Content
        with st.spinner("Loading details..."):
            detail = doc_manager.get_document_detail(source_path)
            
        if detail:
            # Tabs for Chunks and Images
            tab1, tab2 = st.tabs(["Chunks Content", "Images Gallery"])
            
            with tab1:
                for i, chunk in enumerate(detail.chunks):
                    with st.expander(f"Chunk {i+1} (ID: {chunk.id[:8]}...)", expanded=False):
                        st.text_area("Content", chunk.content, height=150, disabled=True)
                        st.json(chunk.metadata)
                        if chunk.images:
                            st.info(f"Contains {len(chunk.images)} images")

            with tab2:
                # Aggregate images
                images_to_show = []
                for chunk in detail.chunks:
                    if chunk.images:
                        for img in chunk.images:
                            images_to_show.append(img)
                
                if images_to_show:
                    cols = st.columns(3)
                    for idx, img_meta in enumerate(images_to_show):
                        img_id = img_meta.get("image_id")
                        caption = img_meta.get("caption", "No caption")
                        
                        if img_id:
                            path = doc_manager.image_storage.get_path(collection=collection, image_id=img_id)
                            with cols[idx % 3]:
                                if path and path.exists():
                                    st.image(str(path), caption=f"{caption} ({img_id})")
                                else:
                                    st.warning(f"Image file not found: {img_id}")
                else:
                    st.info("No images in this document.")
