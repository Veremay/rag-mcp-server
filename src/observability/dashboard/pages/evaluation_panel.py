import os
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List

from src.core.settings import Settings, load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport, EvalCaseResult

# Constants
DEFAULT_TEST_SET_PATH = "tests/fixtures/golden_test_set.json"


def render_evaluation_panel() -> None:
    st.title("Evaluation Panel")
    st.markdown("Run evaluations against golden test sets and visualize metrics.")

    # 1. Configuration Section
    st.header("1. Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_set_path = st.text_input(
            "Golden Test Set Path", 
            value=DEFAULT_TEST_SET_PATH,
            help="Path to the JSON file containing test cases."
        )
        
        # Check if file exists
        if os.path.exists(test_set_path):
            st.success(f"Found test set: {test_set_path}")
            try:
                with open(test_set_path, 'r') as f:
                    data = json.load(f)
                    case_count = len(data.get("test_cases", []))
                    st.info(f"Loaded {case_count} test cases.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.error("File not found.")

    with col2:
        # Load settings to see available backends (though strictly we use what's in settings)
        try:
            settings = load_settings()
            current_backends = settings.evaluation.backends
            st.write("**Current Evaluator Backends:**")
            st.code(str(current_backends), language="json")
            
            top_k = settings.retrieval.top_k_final
            st.write(f"**Retrieval Top-K:** {top_k}")
        except Exception as e:
            st.error(f"Failed to load settings: {e}")
            return

    # 2. Execution Section
    st.header("2. Run Evaluation")
    
    if st.button("🚀 Run Evaluation", type="primary"):
        if not os.path.exists(test_set_path):
            st.error("Please provide a valid test set path.")
            return
            
        with st.spinner("Running evaluation... This may take a while depending on test set size."):
            try:
                # Initialize components freshly to ensure clean state
                settings = load_settings()
                hybrid_search = HybridSearch(settings)
                evaluator = EvaluatorFactory.create(settings)
                runner = EvalRunner(settings, hybrid_search, evaluator)
                
                # Run
                report = runner.run(test_set_path)
                
                # Store results in session state to persist after rerun
                st.session_state["last_eval_report"] = report
                st.success("Evaluation completed!")
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 3. Results Section
    if "last_eval_report" in st.session_state:
        report: EvalReport = st.session_state["last_eval_report"]
        
        st.divider()
        st.header("3. Results Analysis")
        
        # Metrics Summary
        st.subheader("Aggregate Metrics")
        
        # Display metrics in columns
        metrics = report.aggregate_metrics
        cols = st.columns(len(metrics))
        for idx, (k, v) in enumerate(metrics.items()):
            with cols[idx]:
                st.metric(label=k, value=f"{v:.4f}")
        
        # Detailed Results Table
        st.subheader("Detailed Breakdown")
        
        # Convert case results to DataFrame for display
        rows = []
        for case in report.case_results:
            row = {
                "Query": case.query,
                "Metrics": str(case.metrics),  # Flatten for display
                "Retrieved (IDs)": str(case.retrieved_ids),
                "Golden (IDs)": str(case.golden_ids)
            }
            # Add individual metric columns for sorting
            for k, v in case.metrics.items():
                row[k] = v
            rows.append(row)
            
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        # Detailed Inspector
        st.subheader("Case Inspector")
        selected_case_idx = st.selectbox(
            "Select a case to inspect details:",
            range(len(report.case_results)),
            format_func=lambda i: f"[{i+1}] {report.case_results[i].query}"
        )
        
        if selected_case_idx is not None:
            case = report.case_results[selected_case_idx]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Query Info**")
                st.json({
                    "query": case.query,
                    "metrics": case.metrics
                })
                
                st.markdown("**Golden Standard**")
                st.json({
                    "ids": case.golden_ids,
                    "sources": case.golden_sources
                })
                
            with col_b:
                st.markdown("**Retrieved Results**")
                st.json({
                    "ids": case.retrieved_ids,
                    "sources": case.retrieved_sources
                })

