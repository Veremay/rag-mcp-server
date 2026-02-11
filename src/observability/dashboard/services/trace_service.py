import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from src.core.settings import Settings

class TraceService:
    def __init__(self, settings: Settings):
        # In a real app, this might come from settings
        # For now, we assume standard location
        self.log_file = Path("logs/traces.jsonl")
    
    def load_traces(self, trace_type: str = "ingestion", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Load traces of a specific type from the log file.
        Returns a list of dicts, sorted by start time descending.
        """
        if not self.log_file.exists():
            return []
            
        traces = []
        try:
            # Read all lines and filter
            # Optimization for large files: read from end or use robust log reader
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("trace_type") == trace_type:
                            traces.append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            # In a real app, we should log this error
            print(f"Error reading trace logs: {e}")
            return []
            
        # Sort by started_ms descending (newest first)
        traces.sort(key=lambda x: x.get("started_ms", 0), reverse=True)
        return traces[:limit]

    def get_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Find a specific trace by ID."""
        if not self.log_file.exists():
            return None
            
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("trace_id") == trace_id:
                            return data
                    except:
                        continue
        except:
            return None
        return None

    def get_stage_metrics(self, trace: Dict[str, Any]) -> pd.DataFrame:
        """Convert stages to a DataFrame for visualization."""
        stages = trace.get("stages", [])
        if not stages:
            return pd.DataFrame()
            
        data = []
        for stage in stages:
            data.append({
                "Stage": stage.get("name"),
                "Start (ms)": stage.get("start_ms"),
                "End (ms)": stage.get("end_ms"),
                "Duration (ms)": stage.get("duration_ms"),
                "Details": json.dumps(stage.get("data", {}), ensure_ascii=False),
                "Metrics": json.dumps(stage.get("metrics", {}), ensure_ascii=False)
            })
        return pd.DataFrame(data)
