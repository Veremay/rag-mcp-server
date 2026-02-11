from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="start_dashboard")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_project_on_sys_path()

    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    if not _module_available("streamlit"):
        print(
            "ERROR: 未安装 streamlit。请先安装：pip install streamlit", file=sys.stderr
        )
        return 2

    from src.core.settings import load_settings

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path

    try:
        settings = load_settings(str(config_path))
    except Exception as e:
        print(f"ERROR: 加载配置失败：{e}", file=sys.stderr)
        return 1

    port = (
        int(args.port)
        if args.port is not None
        else int(settings.observability.dashboard_port)
    )

    app_path = _project_root() / "src" / "observability" / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"ERROR: Dashboard 文件不存在：{app_path}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    proc = subprocess.run(cmd, cwd=str(_project_root()))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
