import streamlit as st
import pandas as pd
import altair as alt
import json
from datetime import datetime
from src.observability.dashboard.services.trace_service import TraceService
from src.observability.dashboard.services.config_service import ConfigService

def render_query_traces_page() -> None:
    st.header("Query Traces 🔍")
    
    # Initialize services
    config_service = ConfigService()
    settings = config_service.get_settings()
    trace_service = TraceService(settings)
    
    # Load traces
    traces = trace_service.load_traces(trace_type="query", limit=50)
    
    if not traces:
        st.info("No query traces found. Run some queries to generate traces.")
        if st.button("Refresh"):
            st.rerun()
        return

    # Filter
    search_query = st.text_input("Search by query text", "")
    
    filtered_traces = []
    for t in traces:
        # Extract query text
        query_text = "N/A"
        stages = t.get("stages", [])
        for s in stages:
            if s["name"] == "query_processing":
                data = s.get("data", {})
                query_text = data.get("original_query") or data.get("effective_query") or data.get("normalized_query") or "N/A"
                break
        
        t["_query_text"] = query_text # Store for display
        
        if search_query and search_query.lower() not in query_text.lower():
            continue
            
        filtered_traces.append(t)
        
    # Metrics
    total_count = len(filtered_traces)
    avg_duration = sum(t.get("duration_ms", 0) for t in filtered_traces) / total_count if total_count else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Total Queries", total_count)
    m2.metric("Avg Latency", f"{avg_duration:.2f} ms")
    
    st.divider()
    
    st.subheader("History")
    
    # Prepare table data
    table_rows = []
    for t in filtered_traces:
        start_ms = t.get("started_ms", 0)
        dt_str = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract stats
        fusion_hits = 0
        final_hits = 0
        
        stages = t.get("stages", [])
        for s in stages:
            if s["name"] == "fusion":
                fusion_hits = s.get("metrics", {}).get("n_output", 0)
            if s["name"] == "rerank":
                final_hits = s.get("metrics", {}).get("n_output", 0)
        
        # If rerank skipped (e.g. not configured), use fusion output as final
        if final_hits == 0 and fusion_hits > 0:
             # Check if rerank stage exists
             if not any(s["name"] == "rerank" for s in stages):
                 final_hits = fusion_hits
        
        table_rows.append({
            "Trace ID": t["trace_id"],
            "Time": dt_str,
            "Query": t["_query_text"],
            "Duration (ms)": t.get("duration_ms", 0),
            "Retrieved": fusion_hits,
            "Final": final_hits
        })
        
    df = pd.DataFrame(table_rows)
    
    selection = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    selected_trace = None
    if selection.selection.rows:
        idx = selection.selection.rows[0]
        selected_id = df.iloc[idx]["Trace ID"]
        selected_trace = next((t for t in traces if t["trace_id"] == selected_id), None)
    
    if selected_trace:
        st.divider()
        st.subheader(f"Trace Detail")
        st.markdown(f"**Query:** `{selected_trace['_query_text']}`")
        
        # Waterfall Chart
        st.markdown("##### Execution Timeline")
        
        stages_df = trace_service.get_stage_metrics(selected_trace)
        if not stages_df.empty:
            base_time = selected_trace.get("started_ms", 0)
            stages_df["Relative Start"] = stages_df["Start (ms)"] - base_time
            stages_df["Relative End"] = stages_df["End (ms)"] - base_time
            
            chart = alt.Chart(stages_df).mark_bar().encode(
                x=alt.X('Relative Start', title='Time (ms)'),
                x2='Relative End',
                y=alt.Y('Stage', sort=None),
                tooltip=['Stage', 'Duration (ms)', 'Details', 'Metrics']
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
        # Comparison Metrics
        st.markdown("##### Retrieval Pipeline Stats")
        
        # Extract detailed metrics
        dense_hits = 0
        sparse_hits = 0
        fusion_in = 0
        fusion_out = 0
        rerank_in = 0
        rerank_out = 0
        
        for s in selected_trace.get("stages", []):
            metrics = s.get("metrics", {})
            if s["name"] == "dense":
                dense_hits = metrics.get("n_hits", 0)
            elif s["name"] == "sparse":
                sparse_hits = metrics.get("n_hits", 0)
            elif s["name"] == "fusion":
                fusion_in = metrics.get("n_input", 0)
                fusion_out = metrics.get("n_output", 0)
            elif s["name"] == "rerank":
                rerank_in = metrics.get("n_input", 0)
                rerank_out = metrics.get("n_output", 0)
                
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.caption("Retrieval")
            st.metric("Dense Hits", dense_hits)
            st.metric("Sparse Hits", sparse_hits)
            
        with c2:
            st.caption("Fusion")
            st.metric("Input", fusion_in)
            st.metric("Output", fusion_out, delta=fusion_out-fusion_in)
            
        with c3:
            st.caption("Rerank")
            st.metric("Input", rerank_in)
            st.metric("Output", rerank_out, delta=rerank_out-rerank_in)

        with st.expander("Raw JSON"):
            st.json(selected_trace)
