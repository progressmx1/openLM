"""Document attachment/extraction routes.

These are *user-initiated* operations (the user explicitly selects a file), so they
are not subject to the workspace boundary that governs model-generated tool calls.
Extracted files are stored under ``data/documents/``.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..core.paths import get_documents_dir
from ..tools.documents.docx import docx_read
from ..tools.documents.pdf import pdf_extract_text, pdf_metadata
from .deps import AppState, get_state

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_TEXT_CHARS = 50_000


def _kind_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    return "other"


def _extract(saved_path: Path, kind: str) -> dict:
    if kind == "pdf":
        meta = pdf_metadata(saved_path)
        text = "\n\n".join(
            f"--- Page {p['page']} ---\n{p['text']}" for p in pdf_extract_text(saved_path)["pages"]
        )
        return {
            "kind": "pdf",
            "page_count": meta["page_count"],
            "title": meta.get("title"),
            "text": text,
        }
    if kind == "docx":
        data = docx_read(saved_path)
        text = "\n".join(
            b["text"] for b in data["blocks"] if b["type"] in ("paragraph", "heading") and b.get("text")
        )
        return {
            "kind": "docx",
            "paragraph_count": data["paragraph_count"],
            "title": data.get("title"),
            "text": text,
        }
    raise ValueError("Unsupported document type. Only .pdf and .docx are supported.")


@router.post("/extract")
async def extract_document(file: UploadFile, state: AppState = Depends(get_state)) -> dict:
    filename = file.filename or "document"
    kind = _kind_for(filename)
    if kind not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported.")

    documents_dir = get_documents_dir()
    safe_stem = "".join(c for c in Path(filename).stem if c.isalnum() or c in "-_ .") or "doc"
    saved_path = documents_dir / f"{safe_stem}-{uuid.uuid4().hex[:8]}{Path(filename).suffix.lower()}"

    content = await file.read()
    saved_path.write_bytes(content)

    try:
        extracted = _extract(saved_path, kind)
    except Exception as exc:  # noqa: BLE001
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not read document: {exc}")

    text = extracted["text"]
    truncated = len(text) > MAX_TEXT_CHARS
    return {
        "filename": filename,
        "saved_path": str(saved_path),
        "kind": kind,
        "page_count": extracted.get("page_count"),
        "paragraph_count": extracted.get("paragraph_count"),
        "title": extracted.get("title"),
        "text": text[:MAX_TEXT_CHARS],
        "truncated": truncated,
    }


@router.get("")
def list_documents(state: AppState = Depends(get_state)) -> list[dict]:
    documents_dir = get_documents_dir()
    entries = []
    for path in sorted(documents_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_file():
            entries.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                    "kind": _kind_for(path.name),
                }
            )
    return entries
