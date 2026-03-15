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
    
    # 1. Fetch ALL documents to discover available collections
    try:
        all_docs = doc_manager.list_documents() # No collection filter = get all
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return

    # 2. Extract unique collections
    available_collections = sorted(list(set(d.collection for d in all_docs)))
    
    # 3. Determine default selection
    try:
        stats = doc_manager.get_collection_stats()
        default_collection = stats.collection_name
    except Exception:
        default_collection = "knowledge_hub" # Default fallback

    # Ensure default is in the list (so user can see it even if empty)
    if default_collection not in available_collections:
        available_collections.append(default_collection)
        available_collections.sort()
        
    # Find index of default
    try:
        default_index = available_collections.index(default_collection)
    except ValueError:
        default_index = 0

    # 4. Render Selectbox
    collection = st.sidebar.selectbox(
        "Active Collection", 
        options=available_collections, 
        index=default_index,
        help="Select collection to browse"
    )
    
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # 5. Filter docs for display
    # We already have all_docs, just filter in memory
    docs = [d for d in all_docs if d.collection == collection]

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
    st.caption("👉 点击表格中的一行，下方会显示该文档的 Chunk 内容与 Chunk ID（用于构造 ground truth）。")
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
        width='stretch',
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
                st.caption(
                    "构造 ground truth 步骤：① 先看下方每个块的「Chunk 内容」，判断哪些块应该被某 query 命中；"
                    "② 复制对应块的「Chunk ID」到 golden test set 的 expected_chunk_ids；"
                    "source 在 metadata 中可作 expected_sources。"
                )
                for i, chunk in enumerate(detail.chunks):
                    with st.expander("Chunk %d — ID: %s… (%d 字符)" % (i + 1, chunk.id[:12], len(chunk.content)), expanded=False):
                        st.markdown("**Chunk 内容**（据此判断是否应作为某 query 的 ground truth）")
                        st.text_area(
                            "块正文",
                            chunk.content or "(空)",
                            height=200,
                            disabled=True,
                            key="content_%s_%s" % (chunk.id, i),
                            label_visibility="collapsed",
                        )
                        st.markdown("**Chunk ID**（可复制到 golden test set 的 expected_chunk_ids）")
                        st.code(chunk.id, language=None)
                        st.json(chunk.metadata)
                        num_imgs = len(chunk.images) if isinstance(chunk.images, list) else 0
                        if num_imgs:
                            st.info("Contains %d images" % num_imgs)

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
