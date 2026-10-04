from __future__ import annotations

"""Production RAG pipeline combining modules M1 through M5."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build and index the production retrieval pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    started = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    documents = load_documents()
    chunks = []
    for document in documents:
        parents, children = chunk_hierarchical(
            document["text"], metadata=document["metadata"]
        )
        parent_texts = {
            parent.metadata["parent_id"]: parent.text
            for parent in parents
        }
        for child in children:
            chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "parent_text": parent_texts.get(child.parent_id, child.text),
                    "original_text": child.text,
                },
            })
    print(
        f"  ✓ {len(chunks)} child chunks from {len(documents)} documents "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )

    started = time.time()
    print(f"\n[2/4] Enriching {len(chunks)} chunks (M5)...", flush=True)
    enriched = enrich_chunks(chunks)
    if enriched:
        # M5 content is used only for retrieval. Original child and parent text
        # stay in metadata, so generated enrichment cannot leak into answers.
        chunks = [
            {
                "text": item.enriched_text,
                "metadata": {
                    **item.auto_metadata,
                    "original_text": item.original_text,
                },
            }
            for item in enriched
        ]
        print(
            f"  ✓ Enriched {len(enriched)} chunks ({time.time() - started:.1f}s)",
            flush=True,
        )
    else:
        print("  M5 returned no chunks; using raw child chunks", flush=True)

    started = time.time()
    print(f"\n[3/4] Indexing {len(chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(chunks)
    print(f"  ✓ Indexed ({time.time() - started:.1f}s)", flush=True)

    started = time.time()
    print("\n[4/4] Preparing reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time() - started:.1f}s)", flush=True)
    return search, reranker


def _parent_candidates(results) -> list[dict]:
    """Collapse retrieved children into unique, clean parent candidates."""
    candidates = {}
    for result in results:
        metadata = dict(result.metadata)
        clean_text = (
            metadata.get("parent_text")
            or metadata.get("original_text")
            or result.text
        )
        key = (
            metadata.get("source", ""),
            metadata.get("parent_id", clean_text),
        )
        candidate = {
            "text": clean_text,
            "score": float(result.score),
            "metadata": metadata,
        }
        if key not in candidates or candidate["score"] > candidates[key]["score"]:
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda item: item["score"], reverse=True)


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
) -> tuple[str, list[str]]:
    """Retrieve enriched children, rerank clean parents, and answer."""
    retrieved = search.search(query)
    candidates = _parent_candidates(retrieved)
    reranked = reranker.rerank(query, candidates, top_k=RERANK_TOP_K)
    contexts = [result.text for result in reranked]
    if not contexts:
        contexts = [candidate["text"] for candidate in candidates[:RERANK_TOP_K]]

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI

            context_text = "\n\n---\n\n".join(contexts)
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=300,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trả lời trực tiếp và ngắn gọn bằng tiếng Việt, CHỈ dựa trên "
                            "context được cung cấp. Giữ nguyên con số, điều kiện và ngoại lệ. "
                            "Nếu có nhiều phiên bản chính sách, ưu tiên phiên bản mới nhất và "
                            "nói rõ phiên bản cũ đã bị thay thế. Nếu context không đủ, trả lời "
                            "đúng câu: 'Không tìm thấy thông tin.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context_text}\n\n"
                            f"Câu hỏi: {query}\n\nTrả lời trực tiếp:"
                        ),
                    },
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  LLM generation failed: {exc}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run the production pipeline over the evaluation set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    for index, item in enumerate(test_set, start=1):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{index}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    started = time.time()
    print(f"\n[Eval] Running RAGAS on {len(test_set)} questions...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print(f"  ✓ RAGAS done ({time.time() - started:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        score = results.get(metric, 0.0)
        print(f"  {'✓' if score >= 0.75 else '✗'} {metric}: {score:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    pipeline_started = time.time()
    pipeline_search, pipeline_reranker = build_pipeline()
    evaluate_pipeline(pipeline_search, pipeline_reranker)
    print(f"\nTotal: {time.time() - pipeline_started:.1f}s")
