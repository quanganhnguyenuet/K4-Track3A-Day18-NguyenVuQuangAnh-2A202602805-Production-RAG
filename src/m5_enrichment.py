from __future__ import annotations

"""Module 5: optional chunk enrichment with OpenAI and local fallbacks."""

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _openai_json(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> dict | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:
        print(f"  OpenAI enrichment failed: {exc}")
        return None


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def summarize_chunk(text: str) -> str:
    """Return a concise summary, using an extractive fallback offline."""
    if not text.strip():
        return ""
    result = _openai_json(
        "Summarize the supplied Vietnamese text in at most 2 concise sentences. Return JSON with key summary.",
        text,
        max_tokens=150,
    )
    if result and isinstance(result.get("summary"), str):
        return result["summary"].strip()
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2])


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate likely user questions answered by a chunk."""
    if n_questions <= 0 or not text.strip():
        return []
    result = _openai_json(
        f"Generate up to {n_questions} Vietnamese questions answerable from the text. Return JSON with key questions, a list of strings.",
        text,
        max_tokens=220,
    )
    if result and isinstance(result.get("questions"), list):
        questions = [str(q).strip().rstrip("?") + "?" for q in result["questions"] if str(q).strip()]
        if questions:
            return questions[:n_questions]
    sentences = _sentences(text)
    if not sentences:
        return ["Đoạn văn này nói về vấn đề gì?"][:n_questions]
    return [f"Nội dung chính của đoạn này là gì? ({sentence[:100]})?" for sentence in sentences[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Add a short document-context line while preserving the original chunk."""
    if not text.strip():
        return text
    result = _openai_json(
        "Write one concise Vietnamese sentence describing the document context and topic. Return JSON with key context.",
        f"Document: {document_title}\n\nText:\n{text}",
        max_tokens=100,
    )
    context = result.get("context", "").strip() if result else ""
    if not context:
        context = f"Trích từ tài liệu {document_title}." if document_title else "Ngữ cảnh tài liệu."
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract stable metadata using the LLM or deterministic keyword rules."""
    result = _openai_json(
        'Extract metadata and return JSON: {"topic":"...", "entities":[], "category":"policy|hr|it|finance|general", "language":"vi|en"}.',
        text,
        max_tokens=180,
    )
    if result:
        return {
            "topic": str(result.get("topic", "general")),
            "entities": list(result.get("entities", [])) if isinstance(result.get("entities", []), list) else [],
            "category": str(result.get("category", "general")),
            "language": str(result.get("language", "vi")),
        }

    lowered = text.lower()
    if any(word in lowered for word in ("mật khẩu", "vpn", "malware", "mfa", "it ")):
        category = "it"
    elif any(word in lowered for word in ("lương", "nghỉ phép", "nhân viên", "thử việc", "mentor")):
        category = "hr"
    elif any(word in lowered for word in ("chi phí", "ngân sách", "thanh toán", "báo giá")):
        category = "finance"
    else:
        category = "policy"
    entities = re.findall(r"\b(?:PVI|MFA|VPN|WireGuard|AES-?256)\b", text, flags=re.IGNORECASE)
    return {"topic": "general", "entities": sorted(set(entities)), "category": category, "language": "vi"}


def _enrich_single_call(text: str, source: str) -> dict:
    """Return all enrichment fields from one API call, with a local fallback."""
    result = _openai_json(
        """Analyze the supplied chunk for search enrichment. Use only facts stated in the chunk; do not infer or add facts.
Return JSON with keys: summary (a factual 1-2 sentence summary), questions (up to 3 natural Vietnamese questions fully answerable by the chunk), context (one factual sentence identifying the chunk's topic and document location), metadata (object containing topic, entities, category, language).""",
        f"Document: {source}\n\nText:\n{text}",
        max_tokens=400,
    )
    if result:
        return result
    summary = summarize_chunk(text)
    return {
        "summary": summary,
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu {source}." if source else "Ngữ cảnh tài liệu.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks in combined (default) or individually selected modes."""
    methods = ["combined"] if methods is None else list(methods)
    use_combined = "combined" in methods
    enriched = []
    for index, chunk in enumerate(chunks or []):
        text = chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "")
        if use_combined:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", "")).strip()
            questions = [
                str(question).strip()
                for question in result.get("questions", [])
                if str(question).strip()
            ][:3]
            context_line = str(result.get("context", "")).strip()
            retrieval_parts = []
            if context_line:
                retrieval_parts.append(f"Ngữ cảnh: {context_line}")
            if summary and summary != text.strip():
                retrieval_parts.append(f"Tóm tắt: {summary}")
            if questions:
                retrieval_parts.append(
                    "Câu hỏi có thể trả lời:\n- " + "\n- ".join(questions)
                )
            retrieval_parts.append(f"Nội dung gốc: {text}")
            enriched_text = "\n\n".join(retrieval_parts)
            auto_meta = result.get("metadata", {})
            method = "combined"
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
            method = "+".join(methods)
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method=method,
        ))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched


if __name__ == "__main__":
    print(summarize_chunk("Nhân viên được nghỉ phép năm 12 ngày."))
