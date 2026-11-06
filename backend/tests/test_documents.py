"""Tests for PDF and DOCX document tools and the extract endpoint."""
from __future__ import annotations

import io

import pymupdf

from app.tools.documents.docx import ContentBlock, docx_create, docx_read
from app.tools.documents.pdf import (
    pdf_convert,
    pdf_create,
    pdf_detect_scanned,
    pdf_extract_text,
    pdf_metadata,
    pdf_search,
)


def _set_workspace(settings_store, path) -> None:
    settings = settings_store.load()
    settings.approved_workspaces = [str(path)]
    settings_store.save(settings)


# ----------------------------------------------------------------- PDF


def test_pdf_create_and_read(tmp_path) -> None:
    path = tmp_path / "a.pdf"
    pdf_create(path, "Hello world. This is a test.\nSecond paragraph.")

    meta = pdf_metadata(path)
    assert meta["page_count"] == 1

    text = pdf_extract_text(path)
    assert "Hello world" in text["pages"][0]["text"]


def test_pdf_search(tmp_path) -> None:
    path = tmp_path / "b.pdf"
    pdf_create(path, "The quick brown fox jumps over the lazy dog.")
    result = pdf_search(str(path), "fox")
    assert result["count"] >= 1
    assert result["matches"][0]["page"] == 1


def test_pdf_convert_to_txt(tmp_path) -> None:
    path = tmp_path / "c.pdf"
    pdf_create(path, "convert me please")
    out = tmp_path / "c.txt"
    pdf_convert(str(path), "txt", str(out))
    assert out.exists()
    assert "convert me please" in out.read_text(encoding="utf-8")


def test_pdf_detect_scanned_text_pdf(tmp_path) -> None:
    path = tmp_path / "d.pdf"
    pdf_create(path, "has text")
    assert pdf_detect_scanned(str(path))["needs_ocr"] is False


# ----------------------------------------------------------------- DOCX


def test_docx_create_and_read(tmp_path) -> None:
    path = tmp_path / "e.docx"
    blocks = [
        ContentBlock(type="heading", text="Title", level=1),
        ContentBlock(type="paragraph", text="Body text"),
        ContentBlock(type="table", rows=[["a", "b"], ["c", "d"]]),
    ]
    docx_create(path, None, blocks)

    data = docx_read(path)
    assert data["table_count"] == 1
    texts = [b["text"] for b in data["blocks"] if b.get("text")]
    assert "Title" in texts
    assert "Body text" in texts
    table = next(b for b in data["blocks"] if b["type"] == "table")
    assert table["rows"][0] == ["a", "b"]


# ----------------------------------------------------------------- tools


def test_pdf_tool_flow(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    created = tool_executor.submit(
        "pdf_create", {"path": str(ws / "x.pdf"), "content": "Hello PDF", "title": "T"}
    )
    assert created["status"] == "success"

    read = tool_executor.submit("pdf_read", {"path": str(ws / "x.pdf")})
    assert read["status"] == "success"
    assert read["result"]["data"]["page_count"] == 1

    extracted = tool_executor.submit("pdf_extract_text", {"path": str(ws / "x.pdf")})
    assert "Hello PDF" in extracted["result"]["data"]["pages"][0]["text"]


def test_docx_tool_flow(tool_executor, settings_store, tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)

    created = tool_executor.submit(
        "docx_create",
        {"path": str(ws / "x.docx"), "blocks": [{"type": "paragraph", "text": "Hi"}]},
    )
    assert created["status"] == "success"

    read = tool_executor.submit("docx_read", {"path": str(ws / "x.docx")})
    assert read["status"] == "success"


def test_document_tools_are_registered(tool_executor) -> None:
    names = {t["name"] for t in tool_executor.list_tools()}
    assert {"pdf_read", "pdf_create", "pdf_search", "docx_read", "docx_create"} <= names


def test_pdf_ocr_reports_missing_tesseract(tool_executor, settings_store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: None)
    ws = tmp_path / "ws"
    ws.mkdir()
    _set_workspace(settings_store, ws)
    pdf_path = ws / "s.pdf"
    pdf_create(pdf_path, "some text")

    result = tool_executor.submit("pdf_ocr", {"path": str(pdf_path)})
    assert result["status"] == "error"
    assert "Tesseract" in result["result"]["error"]


# ----------------------------------------------------------------- extract endpoint


def _pdf_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_extract_pdf_endpoint(client, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_AI_DOCUMENTS_DIR", str(tmp_path / "docs"))
    response = client.post(
        "/api/documents/extract",
        files={"file": ("sample.pdf", _pdf_bytes("grounded content"), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["kind"] == "pdf"
    assert "grounded content" in data["text"]
    assert data["page_count"] == 1


def test_extract_rejects_unsupported_type(client, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_AI_DOCUMENTS_DIR", str(tmp_path / "docs"))
    response = client.post(
        "/api/documents/extract",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400


def test_list_documents(client, monkeypatch, tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    monkeypatch.setenv("LOCAL_AI_DOCUMENTS_DIR", str(docs_dir))
    docs_dir.mkdir()
    (docs_dir / "sample.pdf").write_bytes(_pdf_bytes("x"))
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 1
