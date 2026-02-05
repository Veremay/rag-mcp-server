from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class BM25Index:
    k1: float
    b: float
    token_pattern: str
    doc_len: Dict[str, int]
    avgdl: float
    postings: Dict[str, Dict[str, float]]
    idf: Dict[str, float]

    def search(self, query: str, top_k: int = 10) -> List[BM25Hit]:
        token_re = re.compile(self.token_pattern, flags=re.UNICODE)
        terms = [t.lower() for t in token_re.findall(query or "")]
        if not terms or not self.doc_len:
            return []

        scores: Dict[str, float] = {}
        n_docs = float(len(self.doc_len))
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            idf = self.idf.get(term)
            if idf is None:
                df = float(len(postings))
                idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in postings.items():
                dl = float(self.doc_len.get(doc_id, 0))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                if denom <= 0:
                    continue
                score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return [
            BM25Hit(chunk_id=doc_id, score=score) for doc_id, score in ranked[:top_k]
        ]


class BM25Indexer:
    def __init__(
        self,
        base_dir: str | Path = "data/db/bm25",
        *,
        token_pattern: str = r"\w+",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._token_pattern = token_pattern
        self._k1 = float(k1)
        self._b = float(b)

    def build(
        self,
        *,
        collection: str,
        chunk_ids: Sequence[str],
        sparse_vectors: Sequence[Mapping[str, float]],
    ) -> BM25Index:
        if len(chunk_ids) != len(sparse_vectors):
            raise ValueError(
                f"chunk_ids and sparse_vectors length mismatch: {len(chunk_ids)} != {len(sparse_vectors)}"
            )

        doc_len: Dict[str, int] = {}
        postings: Dict[str, Dict[str, float]] = {}
        total_len = 0

        for chunk_id, vec in zip(chunk_ids, sparse_vectors, strict=True):
            dl = int(sum(float(v) for v in vec.values()))
            doc_len[chunk_id] = dl
            total_len += dl

            for term, tf in vec.items():
                if not term:
                    continue
                term = str(term).lower()
                postings.setdefault(term, {})[chunk_id] = float(tf)

        avgdl = float(total_len) / float(len(doc_len)) if doc_len else 0.0
        n_docs = float(len(doc_len))

        idf: Dict[str, float] = {}
        for term, posting in postings.items():
            df = float(len(posting))
            idf[term] = math.log1p((n_docs - df + 0.5) / (df + 0.5))

        index = BM25Index(
            k1=self._k1,
            b=self._b,
            token_pattern=self._token_pattern,
            doc_len=doc_len,
            avgdl=avgdl,
            postings=postings,
            idf=idf,
        )
        self.persist(collection=collection, index=index)
        return index

    def persist(self, *, collection: str, index: BM25Index) -> None:
        target_dir = self._base_dir / collection
        target_dir.mkdir(parents=True, exist_ok=True)

        meta_path = target_dir / "meta.json"
        postings_path = target_dir / "postings.json"

        meta = {
            "k1": index.k1,
            "b": index.b,
            "token_pattern": index.token_pattern,
            "doc_len": index.doc_len,
            "avgdl": index.avgdl,
            "idf": index.idf,
        }

        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        postings_path.write_text(
            json.dumps(index.postings, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, *, collection: str) -> BM25Index:
        target_dir = self._base_dir / collection
        meta_path = target_dir / "meta.json"
        postings_path = target_dir / "postings.json"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        postings = json.loads(postings_path.read_text(encoding="utf-8"))

        doc_len = {str(k): int(v) for k, v in (meta.get("doc_len") or {}).items()}
        idf = {str(k): float(v) for k, v in (meta.get("idf") or {}).items()}
        postings_typed: Dict[str, Dict[str, float]] = {}
        for term, docs in (postings or {}).items():
            postings_typed[str(term)] = {str(d): float(tf) for d, tf in docs.items()}

        return BM25Index(
            k1=float(meta["k1"]),
            b=float(meta["b"]),
            token_pattern=str(meta["token_pattern"]),
            doc_len=doc_len,
            avgdl=float(meta["avgdl"]),
            postings=postings_typed,
            idf=idf,
        )

    def search(self, *, collection: str, query: str, top_k: int = 10) -> List[BM25Hit]:
        return self.load(collection=collection).search(query=query, top_k=top_k)


def build_bm25_index(
    *,
    base_dir: str | Path,
    collection: str,
    chunk_ids: Sequence[str],
    sparse_vectors: Sequence[Mapping[str, float]],
    token_pattern: str = r"\w+",
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    return BM25Indexer(base_dir, token_pattern=token_pattern, k1=k1, b=b).build(
        collection=collection, chunk_ids=chunk_ids, sparse_vectors=sparse_vectors
    )
