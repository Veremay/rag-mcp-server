from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

from src.core.settings import Settings, load_settings

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class TraceRow:
    trace_id: str
    finished_ms: Optional[float]
    duration_ms: Optional[float]
    stage_count: int
    trace: JsonDict


def _ms_to_local_time_str(ms: Optional[float]) -> str:
    if ms is None:
        return ""
    dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _iter_tail_lines(path: Path, *, max_lines: int) -> Iterable[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception:
        return []

    lines = text.splitlines()
    if max_lines <= 0:
        return lines
    return lines[-max_lines:]


def _load_trace_rows(log_file: Path, *, max_lines: int) -> List[TraceRow]:
    if not log_file.exists():
        return []

    rows: List[TraceRow] = []
    for raw in reversed(list(_iter_tail_lines(log_file, max_lines=max_lines))):
        line = (raw or "").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        trace_id = str(payload.get("trace_id", "") or "").strip()
        finished_ms = _safe_float(payload.get("finished_ms"))
        duration_ms = _safe_float(payload.get("duration_ms"))
        stages = payload.get("stages") or []
        stage_count = len(stages) if isinstance(stages, list) else 0

        if not trace_id:
            continue

        rows.append(
            TraceRow(
                trace_id=trace_id,
                finished_ms=finished_ms,
                duration_ms=duration_ms,
                stage_count=stage_count,
                trace=payload,
            )
        )

    return rows


def _trace_search_blob(trace: JsonDict) -> str:
    trace_id = str(trace.get("trace_id", "") or "")
    stages = trace.get("stages") or []
    parts: List[str] = [trace_id]
    if isinstance(stages, list):
        for s in stages:
            if not isinstance(s, dict):
                continue
            parts.append(str(s.get("name", "") or ""))
            data = s.get("data")
            if data is not None:
                try:
                    parts.append(json.dumps(data, ensure_ascii=False))
                except Exception:
                    parts.append(str(data))
    return "\n".join(parts).lower()


def _get_settings() -> Settings:
    return load_settings()


def _log_file_from_settings(settings: Settings) -> Path:
    raw = str(getattr(settings.observability, "log_file", "") or "").strip()
    if not raw:
        return Path("logs/traces.jsonl")
    return Path(raw)


def _render_trace_detail(trace: JsonDict) -> None:
    trace_id = str(trace.get("trace_id", "") or "")
    started_ms = _safe_float(trace.get("started_ms"))
    finished_ms = _safe_float(trace.get("finished_ms"))
    duration_ms = _safe_float(trace.get("duration_ms"))

    st.subheader("请求详情")
    st.write(
        {
            "trace_id": trace_id,
            "started_at": _ms_to_local_time_str(started_ms),
            "finished_at": _ms_to_local_time_str(finished_ms),
            "duration_ms": duration_ms,
        }
    )

    stages = trace.get("stages") or []
    if not isinstance(stages, list) or not stages:
        st.info("该 trace 不包含 stages。")
        return

    stage_rows: List[Dict[str, Any]] = []
    for idx, s in enumerate(stages):
        if not isinstance(s, dict):
            continue
        stage_rows.append(
            {
                "idx": idx,
                "name": str(s.get("name", "") or ""),
                "duration_ms": _safe_float(s.get("duration_ms")),
                "start_ms": _safe_float(s.get("start_ms")),
                "end_ms": _safe_float(s.get("end_ms")),
            }
        )

    st.subheader("阶段列表")
    st.dataframe(stage_rows, use_container_width=True, hide_index=True)

    chart_data = [
        {"stage": r["name"], "duration_ms": r["duration_ms"] or 0.0}
        for r in stage_rows
        if r.get("name")
    ]
    if chart_data:
        st.subheader("耗时分布（简版）")
        st.bar_chart(chart_data, x="stage", y="duration_ms", horizontal=True)

    options: List[str] = []
    for r in stage_rows:
        name = str(r.get("name") or "")
        d = r.get("duration_ms")
        options.append(f"{int(r['idx']):02d} {name} ({d} ms)")

    st.subheader("阶段详情")
    selected = st.selectbox("选择一个阶段查看 data/metrics", options=options)
    try:
        selected_idx = int(str(selected).split(" ", 1)[0])
    except Exception:
        selected_idx = 0

    stage = stages[selected_idx] if 0 <= selected_idx < len(stages) else {}
    if not isinstance(stage, dict):
        stage = {}

    st.write(
        {
            "name": stage.get("name"),
            "data": stage.get("data"),
            "metrics": stage.get("metrics"),
        }
    )


def _render_trace_list(rows: Sequence[TraceRow]) -> Optional[TraceRow]:
    st.subheader("请求列表（按时间倒序）")

    table_rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    for r in rows:
        finished_at = _ms_to_local_time_str(r.finished_ms)
        labels.append(
            f"{finished_at} | {r.trace_id[:12]} | {r.duration_ms} ms | stages={r.stage_count}"
        )
        table_rows.append(
            {
                "finished_at": finished_at,
                "trace_id": r.trace_id,
                "duration_ms": r.duration_ms,
                "stages": r.stage_count,
            }
        )

    if table_rows:
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    else:
        st.info("暂无 trace 数据。")

    if not labels:
        return None

    selected = st.selectbox("选择一个 trace 查看详情", options=labels, index=0)
    try:
        selected_index = labels.index(selected)
    except ValueError:
        selected_index = 0
    if 0 <= selected_index < len(rows):
        return rows[selected_index]
    return None


def main() -> None:
    st.set_page_config(page_title="Trace Dashboard", layout="wide")
    st.title("Trace Dashboard")

    settings = _get_settings()
    log_file = _log_file_from_settings(settings)

    with st.sidebar:
        st.header("配置")
        st.write({"log_file": str(log_file)})
        max_lines = int(
            st.number_input("最多读取行数", min_value=100, value=2000, step=100)
        )
        keyword = str(
            st.text_input("Query 关键词筛选（trace_id / stage / data）", value="") or ""
        )
        st.write("修改筛选条件后页面会自动刷新。")

    rows = _load_trace_rows(log_file, max_lines=max_lines)
    if keyword.strip():
        kw = keyword.strip().lower()
        rows = [r for r in rows if kw in _trace_search_blob(r.trace)]

    selected = _render_trace_list(rows)
    if selected is None:
        return

    _render_trace_detail(selected.trace)


if __name__ == "__main__":
    main()
