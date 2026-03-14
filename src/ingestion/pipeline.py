"""
摄取流水线：从文件到向量库/BM25/图片存储的完整流程编排。

按顺序执行：完整性校验(去重) -> 加载 -> 图片入库 -> 切分 -> 变换(精炼/元数据/图注) ->
编码(稠密+稀疏) -> 向量 upsert -> BM25 索引 -> 标记成功。每步带 trace 与 on_progress，
便于观测与跳过未变更文件。

设计上采用「单文件原子」：一次 ingest 只处理一个文件，失败时不会出现「部分写入」；
去重依赖文件 SHA256，避免重复摄取相同内容，同时 force 可覆盖以支持重跑。
"""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchProcessResult
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk, Document
from src.ingestion.storage.bm25_indexer import BM25Index, BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import UpsertResult, VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.deepdoc_pdf_loader import DeepDocPdfLoader
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.observability.logger import write_trace as _write_trace


# ---------------------------------------------------------------------------
# 数据类：用于阶段结果与最终返回值，immutable 避免调用方误改导致与 trace 不一致
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitResult:
    """切分结果：保留原 document 与生成的 chunks，便于单测或只做切分不落库。"""
    document: Document
    chunks: List[Chunk]


@dataclass(frozen=True)
class IngestResult:
    """单次摄取的完整结果；skipped=True 时 document/batch/upsert/bm25 为空，便于调用方区分「跳过」与「完成」。"""
    skipped: bool
    file_hash: str
    document: Optional[Document]
    chunks: List[Chunk]
    batch: Optional[BatchProcessResult]
    upsert: Optional[UpsertResult]
    bm25: Optional[BM25Index]


class IngestionPipeline:
    """
    Minimal ingestion pipeline for MVP.

    This pipeline currently integrates the configured splitter and produces Chunk objects
    from a Document.

    各阶段组件可注入(integrity/loader/transforms/encoder/vector_store/bm25/image_storage)，
    便于测试或替换实现；未注入时在对应步骤内按 settings 或默认链创建，保证开箱即用。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        integrity: Optional[FileIntegrityRegistry] = None,
        loader: Optional[BaseLoader] = None,
        transforms: Optional[Sequence[Any]] = None,
        dense_encoder: Optional[Any] = None,
        sparse_encoder: Optional[Any] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_store: Optional[BaseVectorStore] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            settings: Global application settings.

        未传入的组件在运行时按需解析(如 _resolve_loader、_encode 内创建 encoder)，
        避免构造时强依赖所有后端，同时支持部分替换。
        """
        self._settings = settings
        self._integrity = integrity or FileIntegrityRegistry()
        self._loader = loader
        self._transforms = transforms
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._batch_processor = batch_processor
        self._vector_store = vector_store
        self._bm25_indexer = bm25_indexer
        self._image_storage = image_storage
        # 各组件为 None 时在对应步骤内按 settings 懒创建，这样构造 pipeline 时不必连所有后端，
        # 同时测试可只 mock 需要的那几项（如 loader、vector_store）。

    def ingest(
        self,
        *,
        collection: str,
        file_path: str | Path,
        original_filename: Optional[str] = None,
        force: bool = False,
        trace: Optional[Any] = None,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> IngestResult:
        """
        对单文件执行完整摄取。先算 hash 并判断是否跳过(未 force 且已成功摄取过)，
        再依次执行 load -> 图片处理 -> split -> transform -> encode -> upsert -> bm25 -> finalize。
        trace 用于记录各阶段耗时与数据快照；on_progress 供 UI 展示进度。
        """
        effective_trace: Any = trace if trace is not None else TraceContext(trace_type="ingestion")
        # Ensure trace type is set to ingestion if passed in but empty type
        if isinstance(effective_trace, TraceContext) and effective_trace.trace_type == "query":
             effective_trace.trace_type = "ingestion"
        # 使用局部 record_stage/flush_trace 而非 self 方法，避免把 trace 当实例状态传递，
        # 这样单次 ingest 的 trace 生命周期清晰，且便于在测试里注入不实现 record_stage 的 mock。

        def record_stage(
            name: str,
            *,
            start_ms: float,
            end_ms: float,
            data: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
        ) -> None:
            # 用 getattr + callable 判断，这样传入的 trace 不必一定是 TraceContext，
            # 只要实现 record_stage 即可，方便扩展或测试时用轻量对象。
            fn = getattr(effective_trace, "record_stage", None)
            if not callable(fn):
                return
            fn(
                name,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                data=dict(data or {}),
                metrics=dict(metrics or {}),
            )

        def flush_trace() -> None:
            # 只有 TraceContext 才做 finish + 写入；其他 trace 实现不强制要求 to_dict，避免耦合。
            if not isinstance(effective_trace, TraceContext):
                return
            try:
                # Must finish trace before serializing
                effective_trace.finish()
                _write_trace(effective_trace.to_dict(), settings=self._settings)
            except Exception:
                return

        path = Path(file_path)

        if on_progress:
            on_progress("start", {"path": str(path)})

        # --- 阶段 1：完整性校验 ---
        # 先算 hash 再决定是否跳过，避免重复文件重复做 load/split/encode，节省资源且保持索引一致。
        integrity_start = time.time() * 1000.0
        try:
            file_hash = self._integrity.compute_sha256(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline integrity step failed") from e
        integrity_end = time.time() * 1000.0
        record_stage(
            "integrity",
            start_ms=integrity_start,
            end_ms=integrity_end,
            data={
                "path": str(path),
                "original_filename": original_filename or path.name,
                "hash": file_hash,
            },
        )

        # 未设置 force 且该 hash 已成功摄取过则直接返回，保证幂等且便于 UI 显示「已存在」。
        if not force and self._integrity.should_skip(file_hash):
            if on_progress:
                on_progress("skipped", {"hash": file_hash})
            flush_trace()
            return IngestResult(
                skipped=True,
                file_hash=file_hash,
                document=None,
                chunks=[],
                batch=None,
                upsert=None,
                bm25=None,
            )

        if on_progress:
            on_progress("hash_calculated", {"hash": file_hash})

        # --- 阶段 2：加载文档 ---
        # 按扩展名选择 loader（如 PDF 用 PdfLoader 或 DeepDocPdfLoader），统一得到 Document，
        # 后续阶段只依赖 Document 抽象，不关心具体格式。
        load_start = time.time() * 1000.0
        try:
            loader = self._resolve_loader(path)
            document = loader.load(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline loader step failed") from e
        load_end = time.time() * 1000.0
        record_stage(
            "load",
            start_ms=load_start,
            end_ms=load_end,
            data={
                "doc_id": getattr(document, "id", None),
                "method": loader.__class__.__name__ if loader else "unknown",
            },
        )

        # --- 阶段 3：图片入库 ---
        # 在切分之前处理图片：把提取出的图片迁到按 collection 组织的存储，并更新 document 内引用，
        # 这样后续 split/transform 里看到的已是稳定路径，且多模态检索能正确找到图片。
        # 单张图片失败时 _process_images 内会 catch 并 continue，不抛异常，故不会整文档失败；
        # 只有 _process_images 自身抛出（如 image_refs 异常、或存储不可用等）才会导致本阶段失败。
        image_store_start = time.time() * 1000.0
        try:
            self._process_images(document, collection)
        except Exception as e:
            # If image processing fails, log it but don't fail the whole ingestion?
            # Or should we fail? Given images are critical for multimodal, failing is safer.
            raise RuntimeError("IngestionPipeline image storage step failed") from e
        image_store_end = time.time() * 1000.0
        record_stage(
            "image_store",
            start_ms=image_store_start,
            end_ms=image_store_end,
            data={"count": len(document.metadata.get("images", []))},
            metrics={"count": len(document.metadata.get("images", []))}
        )

        # --- 阶段 4：切分 ---
        # 用配置的 splitter 将 document.text 切成多段，每段带 doc 元数据 + chunk_index，
        # 便于检索时按 collection 过滤以及按顺序还原上下文。
        split_start = time.time() * 1000.0
        try:
            split = self.split(document, trace=effective_trace)
            chunks = split.chunks
            # 每个 chunk 的 metadata 里写入 collection，供向量库/Bm25 按 collection 隔离。
            for c in chunks:
                meta = c.metadata
                if isinstance(meta, dict):
                    meta.setdefault("collection", str(collection))
            if on_progress:
                on_progress("split", {"chunks": chunks})
        except Exception as e:
            raise RuntimeError("IngestionPipeline splitter step failed") from e
        split_end = time.time() * 1000.0
        
        # Capture split preview
        split_data = {"method": self._settings.ingestion.splitter.provider}
        if chunks:
            preview = []
            for c in chunks:
                preview.append({
                    "id": c.id,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "metadata": c.metadata
                })
            split_data["chunks_preview"] = preview

        record_stage(
            "split",
            start_ms=split_start,
            end_ms=split_end,
            data=split_data,
            metrics={"n_chunks": float(len(chunks))},
        )

        # --- 阶段 5：变换 ---
        # 精炼文本、补充元数据、图注等，在编码前完成，这样 embedding 看到的是「最终」文本，
        # 检索质量更好；顺序固定为 Refiner -> Enricher -> Captioner，依赖前一步输出。
        transform_start = time.time() * 1000.0
        try:
            chunks = self._apply_transforms(chunks, trace=effective_trace)
            if on_progress:
                on_progress("transformed", {"chunks": chunks})
        except Exception as e:
            raise RuntimeError("IngestionPipeline transform step failed") from e
        transform_end = time.time() * 1000.0
        
        # Capture transform preview
        transform_data = {"method": "chain", "transforms": [t.__class__.__name__ for t in (self._transforms or [])]}
        if chunks:
            preview = []
            for c in chunks:
                preview.append({
                    "id": c.id,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "metadata": c.metadata
                })
            transform_data["chunks_preview"] = preview

        record_stage(
            "transform",
            start_ms=transform_start,
            end_ms=transform_end,
            data=transform_data,
            metrics={"n_chunks": float(len(chunks))},
        )

        # 再次确保 collection 写入每个 chunk，防止 transform 里替换了 metadata 导致丢失。
        for chunk in chunks:
            chunk.metadata["collection"] = collection

        # --- 阶段 6：编码 ---
        # 稠密向量用于语义检索，稀疏向量用于 BM25 关键词检索；批量处理以控制 API 调用次数与限流。
        encode_start = time.time() * 1000.0
        try:
            batch = self._encode(chunks, trace=effective_trace)
            if on_progress:
                on_progress("encoded", {"batch": batch})
        except Exception as e:
            raise RuntimeError("IngestionPipeline embedding step failed") from e
        encode_end = time.time() * 1000.0
        
        # Capture encode preview
        encode_data = {
            "dense_model": self._settings.embedding.model,
            "sparse_model": "bm25",
        }
        if batch and batch.dense_vectors and len(batch.dense_vectors) > 0:
            encode_data["embedding_dim"] = len(batch.dense_vectors[0])

        record_stage(
            "encode",
            start_ms=encode_start,
            end_ms=encode_end,
            data=encode_data,
            metrics={
                "n_dense": float(len(batch.dense_vectors)),
                "n_sparse": float(len(batch.sparse_vectors)),
            },
        )

        # --- 阶段 7：向量写入 ---
        # 将 chunk 与稠密向量转为 VectorRecord 并写入配置的向量库，支持按 collection 过滤。
        upsert_start = time.time() * 1000.0
        try:
            upsert = self._upsert(chunks, batch.dense_vectors, trace=effective_trace)
            if on_progress:
                on_progress("upserted", {"upsert": upsert})
        except Exception as e:
            raise RuntimeError("IngestionPipeline vector upsert step failed") from e
        upsert_end = time.time() * 1000.0
        
        # Capture upsert preview
        upsert_data = {"method": self._settings.vector_store.backend}
        if upsert and upsert.records:
             upsert_data["upserted_ids"] = [r.id for r in upsert.records]

        record_stage(
            "upsert",
            start_ms=upsert_start,
            end_ms=upsert_end,
            data=upsert_data,
            metrics={"n_records": float(len(upsert.records))},
        )

        # --- 阶段 8：BM25 索引 ---
        # 用稀疏向量更新/创建 BM25 索引，与向量库并存，供混合检索时 SparseRetriever 使用；
        # chunk_ids 与 upsert 结果一致，保证稠密与稀疏检索用的是同一批 chunk。
        bm25_start = time.time() * 1000.0
        try:
            chunk_ids = [r.id for r in upsert.records]
            bm25 = self._build_bm25(
                collection=collection,
                chunk_ids=chunk_ids,
                sparse_vectors=batch.sparse_vectors,
            )
            if on_progress:
                on_progress("bm25_built", {})
        except Exception as e:
            raise RuntimeError("IngestionPipeline bm25 step failed") from e
        bm25_end = time.time() * 1000.0
        record_stage(
            "bm25",
            start_ms=bm25_start,
            end_ms=bm25_end,
            metrics={"n_terms": float(len(getattr(bm25, "postings", []) or []))},
        )

        # --- 阶段 9：收尾 ---
        # 标记该 file_hash 已成功摄取，下次同文件且未 force 时 will be skipped。
        finalize_start = time.time() * 1000.0
        try:
            self._integrity.mark_success(file_hash)
            if on_progress:
                on_progress("complete", {})
        except Exception as e:
            raise RuntimeError("IngestionPipeline finalize step failed") from e
        finalize_end = time.time() * 1000.0
        record_stage(
            "finalize",
            start_ms=finalize_start,
            end_ms=finalize_end,
        )
        flush_trace()

        return IngestResult(
            skipped=False,
            file_hash=file_hash,
            document=document,
            chunks=chunks,
            batch=batch,
            upsert=upsert,
            bm25=bm25,
        )

    def split(self, document: Document, trace: Optional[Any] = None) -> SplitResult:
        """
        Split a Document into Chunks using the configured splitter implementation.

        Args:
            document: Input document to split.
            trace: Optional trace context for observability.

        Returns:
            SplitResult containing the original document and generated chunks.

        通过 SplitterFactory 按 settings 创建 splitter，保证与 ingest 内使用的切分逻辑一致；
        返回 SplitResult 而非仅 chunks，方便测试或「只切分不落库」时仍能拿到原始 document。
        """
        splitter = SplitterFactory.create(self._settings)
        chunk_texts = splitter.split_text(document.text, trace=trace)

        chunks: List[Chunk] = []
        # 继承 document 的 metadata 并加上 chunk_index，便于检索阶段按文档与顺序还原。
        for idx, text in enumerate(chunk_texts):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={**document.metadata, "chunk_index": idx},
                    doc_id=document.id,
                )
            )

        return SplitResult(document=document, chunks=chunks)

    def _resolve_loader(self, path: Path) -> BaseLoader:
        """
        按扩展名选择 loader；注入的 _loader 优先，便于单测或支持更多格式。

        为什么先看 _loader：构造时若注入了自定义 loader，则不再根据扩展名选择，方便测试或
        接入新格式；PDF 再按 settings 的 pdf_parser 在「原始解析」与「DeepDoc」间切换，
        以满足不同 PDF 质量与排版需求。
        """
        if self._loader is not None:
            return self._loader

        ext = path.suffix.lower()
        if ext == ".pdf":
            pdf_parser = getattr(
                self._settings.ingestion.loader, "pdf_parser", "original"
            )
            if pdf_parser == "deepdoc":
                return DeepDocPdfLoader(settings=self._settings.ingestion.loader)
            return PdfLoader(settings=self._settings.ingestion.loader)

        raise ValueError(f"Unsupported file extension: {ext}")

    def _apply_transforms(
        self, chunks: List[Chunk], *, trace: Optional[Any]
    ) -> List[Chunk]:
        """
        按顺序执行 ChunkRefiner -> MetadataEnricher -> ImageCaptioner。
        未注入时使用默认链，保证标题/摘要/图注等元数据在编码前就位。

        顺序不可随意调换：Refiner 先做文本精炼，Enricher 依赖稳定文本补充元数据，
        Captioner 依赖元数据中的图片信息生成图注；若注入自定义链，调用方需保证顺序合理。
        """
        transforms = list(self._transforms) if self._transforms is not None else None
        if transforms is None:
            transforms = [
                ChunkRefiner(self._settings),
                MetadataEnricher(self._settings),
                ImageCaptioner(self._settings),
            ]

        current = chunks
        for t in transforms:
            current = t.transform(current, trace=trace)
        return current

    def _encode(
        self, chunks: List[Chunk], *, trace: Optional[Any]
    ) -> BatchProcessResult:
        """
        稠密+稀疏批量编码。batch_size 默认 10 以兼容部分 API 限制，避免单次请求过大。

        稠密向量用于语义检索、稀疏用于 BM25，两者在同一批 chunk 上生成，保证后续 upsert 与
        BM25 索引使用的 id 一致；未注入时懒创建 encoder/processor，避免 __init__ 依赖具体后端。
        """
        dense_encoder = self._dense_encoder or DenseEncoder(self._settings)
        sparse_encoder = self._sparse_encoder or SparseEncoder()
        # Default batch size reduced to 10 to comply with strict API limits (e.g. SiliconFlow/OpenAI)
        batcher = self._batch_processor or BatchProcessor(batch_size=10)
        return batcher.process(
            chunks,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            trace=trace,
        )

    def _upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        *,
        trace: Optional[Any],
    ) -> UpsertResult:
        """
        将 chunks 与稠密向量转为 VectorRecord 并写入向量库，支持 trace。

        不在此处写稀疏向量：稀疏向量交给 _build_bm25 写入 BM25 索引，向量库只存稠密向量，
        职责分离；VectorUpserter 内部按 settings 选后端，注入 _vector_store 时则用注入实例。
        """
        upserter = VectorUpserter(self._settings, vector_store=self._vector_store)
        return upserter.upsert(chunks, dense_vectors, trace=trace)

    def _build_bm25(
        self,
        *,
        collection: str,
        chunk_ids: Sequence[str],
        sparse_vectors: Sequence[dict[str, float]],
    ) -> BM25Index:
        """
        用稀疏向量更新或创建 BM25 索引，供检索侧 SparseRetriever 使用。

        chunk_ids 与 upsert 返回的 record id 一一对应，保证混合检索时稠密与稀疏结果来自同一批 chunk；
        按 collection 隔离索引，便于多知识库场景。
        """
        indexer = self._bm25_indexer or BM25Indexer()
        return indexer.upsert(
            collection=collection, chunk_ids=chunk_ids, sparse_vectors=sparse_vectors
        )

    def _process_images(self, document: Document, collection: str) -> None:
        """
        Process images extracted from the document:
        1. Move images to ImageStorage (organized by collection).
        2. Update image references in document text and metadata.
        3. Ensure metadata contains detailed image info for Dashboard.

        在 split 之前执行，这样切分后的 chunk 内引用已是稳定存储路径；同时更新 document.text
        中的路径字符串，避免 loader 产生的临时路径在后续环节失效。单张失败仅跳过该张、继续其余，
        若需「全部成功才成功」可在调用方根据 image_refs 长度再判断。
        """
        image_refs = document.metadata.get("image_refs", [])
        if not image_refs:
            return

        storage = self._image_storage or ImageStorage()
        
        # New list for updated references
        new_image_refs = []
        # Detailed image info for Dashboard (List[Dict])
        detailed_images = []
        
        updated_text = document.text
        
        for ref_path in image_refs:
            path_obj = Path(ref_path)
            # Handle potential relative paths (e.g. data/images/...)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
                
            if not path_obj.exists():
                # If absolute path doesn't exist, try relative to CWD
                # loader 可能写出相对路径，resolve 后仍可能不在当前 fs，用 CWD 再试一次以便本地调试。
                if not path_obj.is_absolute():
                     path_obj = Path.cwd() / ref_path

                if not path_obj.exists():
                    continue
                
            image_id = path_obj.stem
            
            try:
                # Add to storage (move file to collection folder)
                stored_path = storage.add_file(
                    file_path=path_obj,
                    collection=collection,
                    image_id=image_id,
                    move=True
                )
                
                # Update reference in text: 将正文里出现的旧路径替换为存储后的路径，便于检索与展示。
                # PdfLoader 可能写出 str(path)，Windows 下与 resolve() 的斜杠形式不同，故同时尝试两种形式。
                original_ref_str = str(ref_path)
                stored_path_str = str(stored_path.resolve())
                
                if original_ref_str in updated_text:
                    updated_text = updated_text.replace(original_ref_str, stored_path_str)
                elif str(path_obj) in updated_text:
                    updated_text = updated_text.replace(str(path_obj), stored_path_str)
                
                new_image_refs.append(stored_path_str)
                detailed_images.append({
                    "image_id": image_id,
                    "path": stored_path_str,
                    "collection": collection,
                    "original_path": str(path_obj)
                })
            except Exception as e:
                # 单张失败不中断整次摄取，只跳过该图并继续，避免因一张坏图导致整文档失败。
                print("Error processing image %s: %s" % (ref_path, e))
                continue
            
        document.text = updated_text
        document.metadata["image_refs"] = new_image_refs
        # Key fix for Dashboard: Set "images" metadata（列表内为 path/collection 等，供前端展示与筛选）
        document.metadata["images"] = detailed_images

    def _store_images(self, *, collection: str, document: Document) -> None:
        """
        将 metadata["images"] 中带 data 字节流的项写入 ImageStorage。
        用于 loader 直接产出 in-memory 图片（如 DeepDoc 解析出的图）而非文件路径的场景；
        当前主流程使用 image_refs + _process_images，此方法保留以备其他入口或兼容。
        """
        images = document.metadata.get("images")
        if not isinstance(images, list) or not images:
            return

        storage = self._image_storage or ImageStorage()
        for item in images:
            if not isinstance(item, dict):
                continue
            image_id = item.get("image_id")
            data = item.get("data")
            ext = item.get("ext") or item.get("extension") or ".png"
            if not isinstance(image_id, str) or not isinstance(
                data, (bytes, bytearray)
            ):
                continue
            storage.save(
                collection=collection, image_id=image_id, data=bytes(data), ext=str(ext)
            )


def split_document(
    settings: Settings, document: Document, trace: Optional[Any] = None
) -> List[Chunk]:
    """
    Convenience function to split a document using the ingestion pipeline.

    Args:
        settings: Global application settings.
        document: Input document to split.
        trace: Optional trace context for observability.

    Returns:
        List of generated chunks.

    仅做切分不落库，供脚本或测试中「只想要 chunks」的场景复用同一套 splitter 配置；
    这样与正式 ingest 的切分行为一致，避免「预览/导出」与「入库」结果不一致。
    """
    return IngestionPipeline(settings).split(document, trace=trace).chunks
