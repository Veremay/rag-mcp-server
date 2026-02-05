from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_project_on_sys_path()

    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path

    try:
        settings = load_settings(str(config_path))
        if hasattr(settings, "vector_store") and hasattr(
            settings.vector_store, "collection_name"
        ):
            settings.vector_store.collection_name = str(args.collection)
        pipeline = IngestionPipeline(settings)
        result = pipeline.ingest(
            collection=str(args.collection),
            file_path=Path(args.path),
            force=bool(args.force),
        )
    except Exception as e:
        _print_exception_chain(e)
        return 1

    if result.skipped:
        print(f"SKIPPED\t{args.path}")
        return 0

    print(f"INGESTED\t{args.path}\tchunks={len(result.chunks)}")
    return 0


def _print_exception_chain(err: BaseException) -> None:
    msgs: list[str] = []
    cur: BaseException | None = err
    while cur is not None:
        msg = str(cur).strip() or cur.__class__.__name__
        if msg not in msgs:
            msgs.append(msg)
        cur = cur.__cause__
    for i, m in enumerate(msgs):
        prefix = "ERROR" if i == 0 else "CAUSE"
        print(f"{prefix}: {m}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
