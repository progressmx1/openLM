"""Tests for the local retrieval (RAG) subsystem."""
from __future__ import annotations

from app.retrieval.keyword import keyword_search
from app.retrieval.service import chunk_text


def test_chunk_text_splits() -> None:
    chunks = chunk_text("first paragraph\n\nsecond paragraph")
    assert len(chunks) >= 1
    assert "first paragraph" in chunks[0]


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []


def test_keyword_search_ranks_matches() -> None:
    chunks = [
        {"text": "the cat sat on the mat", "source_name": "a"},
        {"text": "dogs bark loudly", "source_name": "b"},
    ]
    results = keyword_search(chunks, "cat")
    assert results
    assert "cat" in results[0]["text"]


def test_keyword_search_no_match() -> None:
    chunks = [{"text": "nothing relevant here"}]
    assert keyword_search(chunks, "zzzquux") == []


def test_index_and_search(retrieval) -> None:
    retrieval.index_text("doc1", "The capital of France is Paris.")
    results = retrieval.search("capital France")
    assert any("Paris" in r["text"] for r in results)


def test_index_folder(retrieval, tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("alpha beta gamma delta", encoding="utf-8")
    result = retrieval.index_folder(str(tmp_path))
    assert result["indexed"] >= 1
    sources = retrieval.list_sources()
    assert any(s["name"] == "notes.txt" for s in sources)


def test_delete_source(retrieval) -> None:
    source = retrieval.index_text("doc", "some unique content")
    retrieval.delete_source(source["source_id"])
    assert retrieval.list_sources() == []
