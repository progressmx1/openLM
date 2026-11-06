"""Retrieval service: index local files/folders and search them offline.

Uses a keyword (BM25-lite) backend by default so it works with zero extra
dependencies. The design keeps the retrieval backend pluggable: a local
embeddings/vector backend can replace :func:`retrieval.keyword.keyword_search`
without changing the service or API.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from ..storage.repository import Repository
from ..tools.documents.docx import docx_read
from ..tools.documents.pdf import pdf_extract_text
from .keyword import keyword_search

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".html", ".htm",
    ".css", ".scss", ".sh", ".bat", ".ps1", ".c", ".h", ".cpp", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".sql", ".xml", ".toml", ".ini",
    ".cfg", ".log",
}

CHUNK_SIZE = 800
MAX_FILE_BYTES = 2_000_000


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks on paragraph boundaries, falling back to hard cuts."""
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Hard-split an oversized paragraph.
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size:]
            current = para
    if current:
        chunks.append(current)
    return chunks


class RetrievalService:
    def __init__(self, repository: Repository) -> None:
        self.repo = repository

    def index_text(self, name: str, text: str, kind: str = "text", path: Optional[str] = None) -> dict[str, Any]:
        chunks = chunk_text(text)
        source_id = uuid.uuid4().hex
        self.repo.add_rag_source(source_id, name, kind, path)
        self.repo.add_rag_chunks(source_id, chunks)
        return {"source_id": source_id, "name": name, "chunks": len(chunks)}

    def index_folder(self, folder: str) -> dict[str, Any]:
        root = Path(folder)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        indexed = 0
        skipped = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = self._extract_file(path)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if not text or not text.strip():
                skipped += 1
                continue
            self.index_text(path.name, text, kind=path.suffix.lstrip(".") or "text", path=str(path))
            indexed += 1
        return {"indexed": indexed, "skipped": skipped, "folder": str(root)}

    def _extract_file(self, path: Path) -> Optional[str]:
        suffix = path.suffix.lower()
        if suffix in (".pdf",):
            pages = pdf_extract_text(str(path))["pages"]
            return "\n\n".join(p["text"] for p in pages)
        if suffix in (".docx",):
            data = docx_read(str(path))
            return "\n".join(b["text"] for b in data["blocks"] if b.get("text"))
        if suffix in TEXT_EXTENSIONS:
            if path.stat().st_size > MAX_FILE_BYTES:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        return None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        chunks = self.repo.list_rag_chunks()
        matches = keyword_search(chunks, query, top_k)
        return [
            {
                "source": m.get("source_name"),
                "source_id": m.get("source_id"),
                "text": (m.get("text") or "")[:500],
            }
            for m in matches
        ]

    def list_sources(self) -> list[dict[str, Any]]:
        return self.repo.list_rag_sources()

    def delete_source(self, source_id: str) -> bool:
        self.repo.delete_rag_source(source_id)
        return True
