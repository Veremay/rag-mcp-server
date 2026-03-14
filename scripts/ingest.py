from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_project_on_sys_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="一键 ingest：可指定单个 PDF 或包含多个 PDF 的目录（递归处理）。",
    )
    parser.add_argument("--collection", required=True, help="向量集合名称")
    parser.add_argument(
        "--path",
        required=True,
        help="单个 PDF 文件路径，或包含多个 PDF 的目录路径（会递归收集该目录下所有 .pdf 并依次 ingest）",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    return parser.parse_args(argv)


def _format_snippet(text: str, max_len: int = 60) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max(0, max_len - 1)] + "…"


# 与 pipeline._resolve_loader 支持的扩展保持一致，避免把目录当文件打开导致 Permission denied
_SUPPORTED_EXTENSIONS = (".pdf",)


def _collect_file_paths(path: Path) -> list[Path]:
    """
    若 path 为文件且扩展名支持则返回 [path]；
    若为目录则递归收集支持扩展名的文件列表。
    这样传入目录时不会把目录路径交给 pipeline（否则 compute_sha256 会 Permission denied）。
    """
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            return [path]
        return []
    if path.is_dir():
        return [
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    _ensure_project_on_sys_path()

    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])

    from src.core.settings import load_settings
    from src.ingestion.pipeline import IngestionPipeline

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _project_root() / config_path

    # Define progress callback
    def print_progress(stage: str, data: Dict[str, Any]) -> None:
        if not args.verbose:
            return

        if stage == "start":
            print(f"🚀  START Ingest: {data['path']}")
        elif stage == "skipped":
            print(f"⏭️   SKIPPED (Hash match): {data['hash'][:8]}...")
        elif stage == "hash_calculated":
            print(f"🔒  File Hash: {data['hash']}")
        elif stage == "loaded":
            doc = data.get("document")
            meta = doc.metadata if doc else {}
            print(f"📄  LOADED Document: ID={doc.id if doc else '?'} Metadata={meta}")
        elif stage == "split":
            chunks = data.get("chunks", [])
            print(f"✂️   SPLIT into {len(chunks)} chunks")
            for i, c in enumerate(chunks[:3]):
                print(f"    - Chunk {i}: {_format_snippet(c.text)}")
            if len(chunks) > 3:
                print(f"    - ... and {len(chunks) - 3} more")
        elif stage == "transformed":
            chunks = data.get("chunks", [])
            print(
                f"🔄  TRANSFORMED {len(chunks)} chunks (Refinement/Captioning/Enrichment)"
            )
        elif stage == "encoded":
            batch = data.get("batch")
            dense_vecs = batch.dense_vectors if batch else []
            sparse_vecs = batch.sparse_vectors if batch else []
            dim = len(dense_vecs[0]) if dense_vecs else 0
            print(
                f"🧠  ENCODED: Dense[{len(dense_vecs)}x{dim}], Sparse[{len(sparse_vecs)}]"
            )
        elif stage == "upserted":
            upsert = data.get("upsert")
            count = len(upsert.records) if upsert else 0
            print(f"💾  UPSERTED {count} vectors to Store")
        elif stage == "bm25_built":
            print(f"📚  BM25 Index Updated")
        elif stage == "complete":
            print(f"✅  COMPLETE")

    path_arg = Path(args.path).resolve()
    files = _collect_file_paths(path_arg)
    if not files:
        if path_arg.is_dir():
            print(
                "ERROR: 目录下未找到支持的文件（当前支持: %s）"
                % ", ".join(_SUPPORTED_EXTENSIONS),
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: 路径不是文件或扩展名不支持（支持: %s）"
                % ", ".join(_SUPPORTED_EXTENSIONS),
                file=sys.stderr,
            )
        return 1

    try:
        settings = load_settings(str(config_path))
        if hasattr(settings, "vector_store") and hasattr(
            settings.vector_store, "collection_name"
        ):
            settings.vector_store.collection_name = str(args.collection)
        pipeline = IngestionPipeline(settings)
    except Exception as e:
        _print_exception_chain(e)
        return 1

    exit_code = 0
    for one_path in files:
        try:
            result = pipeline.ingest(
                collection=str(args.collection),
                file_path=one_path,
                force=bool(args.force),
                on_progress=print_progress,
            )
        except Exception as e:
            _print_exception_chain(e)
            exit_code = 1
            continue

        if not args.verbose:
            if result.skipped:
                print("SKIPPED\t%s" % one_path)
            else:
                print("INGESTED\t%s\tchunks=%s" % (one_path, len(result.chunks)))

        if len(result.chunks) == 0 and not result.skipped:
            print(
                "WARN: 文档未抽取到可切分的文本，因此 chunks=0。",
                file=sys.stderr,
            )
            print(
                "      常见原因：PDF 为扫描件/图片型；或 PDF 加密/字体编码导致 pypdf 抽取失败。",
                file=sys.stderr,
            )
            print(
                "      建议：先对 PDF 做 OCR（生成文本层）后再运行 ingest.py；或更换为可搜索 PDF。",
                file=sys.stderr,
            )

    return exit_code


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
