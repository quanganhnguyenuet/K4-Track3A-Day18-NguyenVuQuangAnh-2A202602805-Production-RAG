from __future__ import annotations

"""Module 4: RAGAS evaluation and bottom-N failure analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _zero_results(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question = [EvalResult(
        question=question,
        answer=answers[index] if index < len(answers) else "",
        contexts=contexts[index] if index < len(contexts) else [],
        ground_truth=ground_truths[index] if index < len(ground_truths) else "",
        faithfulness=0.0,
        answer_relevancy=0.0,
        context_precision=0.0,
        context_recall=0.0,
    ) for index, question in enumerate(questions)]
    return {**{metric: 0.0 for metric in METRICS}, "per_question": per_question}


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS when configured, returning a stable fallback otherwise."""
    if not questions or not OPENAI_API_KEY:
        return _zero_results(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            value = lambda name: float(row.get(name, 0.0)) if not _is_nan(row.get(name, 0.0)) else 0.0
            per_question.append(EvalResult(
                question=questions[index] if index < len(questions) else "",
                answer=answers[index] if index < len(answers) else "",
                contexts=contexts[index] if index < len(contexts) else [],
                ground_truth=ground_truths[index] if index < len(ground_truths) else "",
                faithfulness=value("faithfulness"),
                answer_relevancy=value("answer_relevancy"),
                context_precision=value("context_precision"),
                context_recall=value("context_recall"),
            ))
        if not per_question:
            return _zero_results(questions, answers, contexts, ground_truths)
        return {
            metric: sum(getattr(item, metric) for item in per_question) / len(per_question)
            for metric in METRICS
        } | {"per_question": per_question}
    except Exception as exc:
        print(f"  RAGAS evaluation failed: {exc}")
        return _zero_results(questions, answers, contexts, ground_truths)


def _is_nan(value) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Return the lowest-scoring questions with a diagnostic recommendation."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "Answer contains claims unsupported by the retrieved context",
            "Tighten the grounded-answer prompt and improve context filtering",
        ),
        "context_recall": (
            "Relevant information was not retrieved",
            "Improve chunking, query expansion, or add BM25 candidates",
        ),
        "context_precision": (
            "Retrieved context contains too many irrelevant chunks",
            "Improve reranking or add metadata filters",
        ),
        "answer_relevancy": (
            "Answer does not directly address the question",
            "Improve the answer prompt and require a concise direct response",
        ),
    }
    scored = []
    for item in eval_results or []:
        if isinstance(item, dict):
            item = EvalResult(**{key: item.get(key, [] if key == "contexts" else "") for key in EvalResult.__dataclass_fields__})
        values = {metric: float(getattr(item, metric, 0.0)) for metric in METRICS}
        worst_metric = min(values, key=values.get)
        scored.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": values[worst_metric],
            "average_score": sum(values.values()) / len(values),
            "diagnosis": diagnostic_tree[worst_metric][0],
            "suggested_fix": diagnostic_tree[worst_metric][1],
        })
    scored.sort(key=lambda item: item["average_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {key: value for key, value in results.items() if key != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=asdict)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
