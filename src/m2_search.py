from __future__ import annotations

"""Module 2: Hybrid BM25 + dense search with reciprocal-rank fusion."""

import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    BM25_TOP_K,
    DENSE_TOP_K,
    HYBRID_TOP_K,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", or "hybrid"


def segment_vietnamese(text: str) -> str:
    """Return whitespace-tokenized Vietnamese text suitable for BM25."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(str(text), format="text")
    except Exception:
        segmented = str(text)
    return " ".join(segmented.replace("_", " ").split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25Okapi index from ``{"text", "metadata"}`` chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks or [])
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).split()
            for chunk in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search BM25 results, omitting documents with zero lexical score."""
        if self.bm25 is None or not query or top_k <= 0:
            return []
        query_tokens = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results: list[SearchResult] = []
        for index, score in ranked[:top_k]:
            score = float(score)
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(SearchResult(
                text=document.get("text", ""),
                score=score,
                metadata=dict(document.get("metadata", {})),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Encode chunks and replace the configured Qdrant collection."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        chunks = list(chunks or [])
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = dict(chunk.get("metadata", {}))
            payload["text"] = chunk.get("text", "")
            points.append(PointStruct(id=index, vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Query Qdrant using its current ``query_points`` API."""
        if not query or top_k <= 0:
            return []
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            )
        except Exception:
            return []

        results = []
        for point in getattr(response, "points", []) or []:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(
                text=text,
                score=float(point.score),
                metadata=payload,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists with ``score += 1 / (k + rank)``."""
    if top_k <= 0 or k < 0:
        return []
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results or [], start=1):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank)

    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]


class HybridSearch:
    """Combine BM25 and dense retrieval with reciprocal-rank fusion."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original: {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
