"""Keyword (BM25-lite) retrieval — a dependency-free, offline retriever.

This is the default retrieval backend and works with no embeddings model. A
vector/embedding backend can be plugged in by implementing the same
``search(chunks, query, top_k)`` surface; see ``retrieval/service.py``.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _score_document(doc_tokens: list[str], query_tokens: list[str], df: Counter, N: int, avgdl: float) -> float:
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    k1, b = 1.5, 0.75
    score = 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        doc_freq = df[term]
        idf = math.log(1 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
        norm_tf = tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * doc_len / max(avgdl, 1.0)))
        score += idf * norm_tf
    return score


def keyword_search(chunks: list[dict[str, Any]], query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Score and rank ``chunks`` (each with a ``text`` field) for ``query``."""
    if not chunks:
        return []
    doc_tokens = [tokenize(c.get("text", "")) for c in chunks]
    df: Counter = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1
    avgdl = sum(len(t) for t in doc_tokens) / len(doc_tokens)
    query_tokens = tokenize(query)

    scored = []
    for chunk, tokens in zip(chunks, doc_tokens):
        score = _score_document(tokens, query_tokens, df, len(chunks), avgdl)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: -pair[0])
    return [chunk for _score, chunk in scored[:top_k]]
