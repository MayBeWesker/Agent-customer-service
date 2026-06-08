from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


def tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", " ", text.lower())
    terms = [token for token in normalized.split() if token]
    if any("\u4e00" <= ch <= "\u9fa5" for ch in normalized):
        chinese_chars = [ch for ch in normalized if "\u4e00" <= ch <= "\u9fa5"]
        for idx in range(len(chinese_chars) - 1):
            terms.append("".join(chinese_chars[idx:idx + 2]))
    return terms


@dataclass
class SearchResult:
    score: float
    payload: dict


class VectorStoreService:
    """A lightweight lexical retriever that mirrors the original vector-store role."""

    def __init__(self, documents: Iterable[dict] | None = None):
        self.documents: list[dict] = []
        self.doc_vectors: list[Counter[str]] = []
        if documents:
            self.add_documents(documents)

    def add_documents(self, documents: Iterable[dict]) -> None:
        for document in documents:
            combined = " ".join(
                str(document.get(field, ""))
                for field in ("name", "aliases", "summary", "usage", "side_effects", "contraindications", "missed_dose", "interactions")
            )
            vector = Counter(tokenize(combined))
            self.documents.append(document)
            self.doc_vectors.append(vector)

    def query(self, question: str, top_k: int = 3) -> list[SearchResult]:
        q_vector = Counter(tokenize(question))
        scored: list[SearchResult] = []
        for document, d_vector in zip(self.documents, self.doc_vectors):
            score = self._cosine_similarity(q_vector, d_vector)
            if score > 0:
                scored.append(SearchResult(score=score, payload=document))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)
