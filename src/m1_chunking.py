from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────



def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None
) -> list[Chunk]:
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm
    import re

    metadata = metadata or {}

    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+|\n\n', text)
        if s.strip()
    ]

    if not sentences:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    cosine_sim = lambda a, b: (
        dot(a, b) / (norm(a) * norm(b) + 1e-9)
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        if cosine_sim(embeddings[i - 1], embeddings[i]) < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_chunk),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "strategy": "semantic"
                    }
                )
            )

            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(
            Chunk(
                text=" ".join(current_chunk),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "strategy": "semantic"
                }
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────



def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None
) -> tuple[list[Chunk], list[Chunk]]:

    if parent_size <= 0 or child_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    metadata = metadata or {}

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    parents = []
    children = []

    # Chia đoạn quá dài, ưu tiên vị trí khoảng trắng
    def split_to_size(content, size):
        parts = []
        content = content.strip()

        while len(content) > size:
            cut = content.rfind(" ", 0, size + 1)

            if cut <= 0:
                cut = size

            parts.append(content[:cut].strip())
            content = content[cut:].strip()

        if content:
            parts.append(content)

        return [p for p in parts if p]

    # Tạo parent và các children tương ứng
    def save_parent(parent_text):
        pid = f"parent_{len(parents)}"

        parents.append(
            Chunk(
                text=parent_text,
                metadata={
                    **metadata,
                    "chunk_type": "parent",
                    "parent_id": pid
                }
            )
        )

        for child_text in split_to_size(
            parent_text, child_size
        ):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child"
                    },
                    parent_id=pid
                )
            )

    # Gom paragraphs thành parents
    current_parent = ""

    for para in paragraphs:
        for part in split_to_size(para, parent_size):

            separator = "\n\n" if current_parent else ""

            if (
                current_parent
                and len(current_parent) + len(separator)
                + len(part) > parent_size
            ):
                save_parent(current_parent)
                current_parent = ""
                separator = ""

            current_parent += separator + part

    # Lưu parent cuối
    if current_parent:
        save_parent(current_parent)

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────



def chunk_structure_aware(
    text: str,
    metadata: dict | None = None
) -> list[Chunk]:
    import re

    metadata = metadata or {}

    sections = re.split(
        r'(^#{1,3}\s+.+$)',
        text,
        flags=re.MULTILINE
    )

    chunks = []

    # Xử lý nội dung trước header đầu tiên
    preamble = sections[0].strip()

    if preamble:
        chunks.append(
            Chunk(
                text=preamble,
                metadata={
                    **metadata,
                    "section": "",
                    "strategy": "structure"
                }
            )
        )

    # Duyệt các cặp header và content
    for i in range(1, len(sections), 2):
        header = sections[i].strip()

        content = (
            sections[i + 1].strip()
            if i + 1 < len(sections)
            else ""
        )

        chunk_text = f"{header}\n\n{content}".strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "section": header,
                        "strategy": "structure"
                    }
                )
            )

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
