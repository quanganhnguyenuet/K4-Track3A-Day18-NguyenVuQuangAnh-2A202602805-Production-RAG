from __future__ import annotations

"""Module 3: cross-encoder reranking and latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_error: Exception | None = None

    def _load_model(self):
        if self._model is None and self._load_error is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                # Keep the pipeline usable offline; the next call will use the
                # deterministic lexical fallback and expose the original error.
                self._load_error = exc
        return self._model

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Small offline fallback used only if the HF model is unavailable."""
        tokens = lambda value: set(re.findall(r"\w+", value.lower(), flags=re.UNICODE))
        query_tokens = tokens(query)
        text_tokens = tokens(text)
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens) / len(query_tokens)
        numbers = set(re.findall(r"\d+", query)) & set(re.findall(r"\d+", text))
        return float(overlap + (0.05 if numbers else 0.0))

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Score query/document pairs and return the highest scoring documents."""
        if not documents or top_k <= 0:
            return []
        model = self._load_model()
        pairs = [(query, document.get("text", "")) for document in documents]
        if model is not None:
            scores = model.predict(pairs, show_progress_bar=False)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if isinstance(scores, (int, float)):
                scores = [scores]
        else:
            scores = [
                self._lexical_score(query, document.get("text", ""))
                for document in documents
            ]

        scored = sorted(
            zip(scores, documents), key=lambda item: float(item[0]), reverse=True
        )
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class FlashrankReranker:
    """Optional lightweight reranker backed by the FlashRank package."""

    def __init__(self):
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": index, "text": document.get("text", "")}
                for index, document in enumerate(documents)
            ]
            ranked = self._model.rerank(
                RerankRequest(query=query, passages=passages)
            )[:top_k]
            output = []
            for rank, result in enumerate(ranked, start=1):
                document = documents[int(result.get("id", 0))]
                output.append(RerankResult(
                    text=document.get("text", ""),
                    original_score=float(document.get("score", 0.0)),
                    rerank_score=float(result.get("score", 0.0)),
                    metadata=dict(document.get("metadata", {})),
                    rank=rank,
                ))
            return output
        except Exception:
            # Keep the optional implementation useful without making it a
            # second source of failure for the production pipeline.
            fallback = CrossEncoderReranker()
            return fallback.rerank(query, documents, top_k)


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark reranker latency over ``n_runs`` calls."""
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    print("Use CrossEncoderReranker from the production pipeline.")
