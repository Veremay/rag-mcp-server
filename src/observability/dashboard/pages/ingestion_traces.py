import streamlit as st
import pandas as pd
import altair as alt
import json
from datetime import datetime
from src.observability.dashboard.services.trace_service import TraceService
from src.observability.dashboard.services.config_service import ConfigService

def render_ingestion_traces_page() -> None:
    st.header("Ingestion Traces 🕵️")
    
    # Initialize services
    config_service = ConfigService()
    settings = config_service.get_settings()
    trace_service = TraceService(settings)
    
    # Load traces
    traces = trace_service.load_traces(trace_type="ingestion", limit=50)
    
    if not traces:
        st.info("No ingestion traces found. Run ingestion to generate traces.")
        if st.button("Refresh"):
            st.rerun()
        return

    # Metrics
    total_count = len(traces)
    avg_duration = sum(t.get("duration_ms", 0) for t in traces) / total_count if total_count else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Total Ingestions (Last 50)", total_count)
    m2.metric("Avg Duration", f"{avg_duration:.2f} ms")
    
    st.divider()
    
    st.subheader("History")
    
    # Prepare table data
    table_rows = []
    for t in traces:
        start_ms = t.get("started_ms", 0)
        dt_str = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate stats from stages
        n_chunks = 0
        file_name = "N/A"
        
        stages = t.get("stages", [])
        for s in stages:
            if s["name"] == "integrity":
                data = s.get("data", {})
                file_name = data.get("original_filename")
                if not file_name:
                    path = data.get("path", "")
                    if path:
                        file_name = path.split("/")[-1]

            if s["name"] == "split" and "n_chunks" in s.get("metrics", {}):
                n_chunks = s["metrics"]["n_chunks"]
        
        table_rows.append({
            "Trace ID": t["trace_id"],
            "Time": dt_str,
            "Duration (ms)": t.get("duration_ms", 0),
            "File": file_name,
            "Chunks": n_chunks,
            "Status": "✅" if t.get("finished_ms") else "❌"
        })
        
    df = pd.DataFrame(table_rows)
    
    # Display dataframe with selection
    # Note: on_select is available in newer Streamlit versions. 
    # If not available, we might need a fallback, but assuming recent version.
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
        st.subheader(f"Trace Detail: {selected_trace['trace_id']}")
        
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
            
            st.markdown("##### Stage Details")
            st.dataframe(
                stages_df[["Stage", "Duration (ms)", "Details", "Metrics"]], 
                use_container_width=True,
                hide_index=True
            )
            
        with st.expander("Raw JSON"):
            st.json(selected_trace)
