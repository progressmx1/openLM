"""DOCX operations built on python-docx.

Like the PDF module, pure functions operate on plain paths (for the user-initiated
extract endpoint) while ``Tool`` wrappers enforce workspace boundaries.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, Optional

from docx import Document
from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel


class ContentBlock(BaseModel):
    type: Literal["paragraph", "heading", "table"]
    text: Optional[str] = None
    level: int = Field(default=1, ge=1, le=9)
    rows: Optional[list[list[str]]] = None


def docx_read(path: str | Path) -> dict[str, Any]:
    doc = Document(str(path))
    core = doc.core_properties
    blocks: list[dict[str, Any]] = []
    for para in doc.paragraphs:
        style = (para.style.name or "").lower()
        if style.startswith("heading") or style == "title":
            btype = "heading"
        elif style.startswith("list"):
            btype = "list"
        else:
            btype = "paragraph"
        if para.text.strip():
            blocks.append({"type": btype, "text": para.text, "style": para.style.name})
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        blocks.append({"type": "table", "rows": rows})

    return {
        "path": str(Path(path).resolve()),
        "title": core.title,
        "author": core.author,
        "paragraph_count": len(doc.paragraphs),
        "table_count": len(doc.tables),
        "blocks": blocks,
    }


def _add_blocks(doc: Document, blocks: list[ContentBlock]) -> None:
    for block in blocks:
        if block.type == "heading":
            doc.add_heading(block.text or "", level=block.level)
        elif block.type == "table" and block.rows:
            rows = block.rows
            cols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=cols)
            table.style = "Table Grid"
            for i, row in enumerate(rows):
                for j in range(cols):
                    table.cell(i, j).text = row[j] if j < len(row) else ""
        else:
            doc.add_paragraph(block.text or "")


def docx_create(
    path: str | Path, title: Optional[str], blocks: list[ContentBlock]
) -> dict[str, Any]:
    doc = Document()
    if title:
        doc.add_heading(title, level=0)
        doc.core_properties.title = title
    _add_blocks(doc, blocks)
    doc.save(str(path))
    return {"path": str(Path(path).resolve())}


def docx_append(path: str | Path, blocks: list[ContentBlock]) -> dict[str, Any]:
    doc = Document(str(path))
    _add_blocks(doc, blocks)
    doc.save(str(path))
    return {"path": str(Path(path).resolve())}


def docx_create_from_text(path: str | Path, text: str) -> dict[str, Any]:
    doc = Document()
    for line in text.split("\n"):
        if line.strip():
            doc.add_paragraph(line)
    doc.save(str(path))
    return {"path": str(Path(path).resolve())}


def _find_soffice() -> Optional[str]:
    return shutil.which("soffice") or shutil.which("soffice.exe") or shutil.which("libreoffice")


def docx_export_pdf(path: str | Path, output: str | Path) -> dict[str, Any]:
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError(
            "DOCX→PDF conversion requires LibreOffice (soffice), which was not found on this system."
        )
    out_dir = str(Path(output).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, str(path)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    produced = Path(out_dir) / (Path(path).stem + ".pdf")
    if not produced.exists():
        raise RuntimeError("LibreOffice did not produce a PDF.")
    if Path(output).resolve() != produced.resolve():
        shutil.move(str(produced), str(output))
    return {"path": str(Path(output).resolve())}


# ------------------------------------------------------------------ tools

class DocxReadRequest(BaseModel):
    path: str


class DocxCreateRequest(BaseModel):
    path: str
    title: Optional[str] = None
    blocks: list[ContentBlock] = Field(default_factory=list)


class DocxAppendRequest(BaseModel):
    path: str
    blocks: list[ContentBlock]


class DocxExportPdfRequest(BaseModel):
    path: str
    output: str


def register_docx_tools(registry: ToolRegistry) -> None:
    def _read(ctx: ExecutionContext, args: DocxReadRequest) -> ToolResult:
        try:
            return ToolResult("success", data=docx_read(ctx.resolve_path(args.path)))
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _create(ctx: ExecutionContext, args: DocxCreateRequest) -> ToolResult:
        try:
            return ToolResult("success", data=docx_create(ctx.resolve_path(args.path), args.title, args.blocks))
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _append(ctx: ExecutionContext, args: DocxAppendRequest) -> ToolResult:
        try:
            return ToolResult("success", data=docx_append(ctx.resolve_path(args.path), args.blocks))
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _export(ctx: ExecutionContext, args: DocxExportPdfRequest) -> ToolResult:
        try:
            return ToolResult(
                "success",
                data=docx_export_pdf(ctx.resolve_path(args.path), ctx.resolve_path(args.output)),
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    registry.register(Tool("docx_read", "Read DOCX paragraphs, headings, tables, and metadata.", PermissionLevel.READ_ONLY, DocxReadRequest, ("path",), handler=_read))
    registry.register(Tool("docx_create", "Create a DOCX document with paragraphs, headings, and tables.", PermissionLevel.WORKSPACE_WRITE, DocxCreateRequest, ("path",), handler=_create))
    registry.register(Tool("docx_append", "Append paragraphs, headings, or tables to a DOCX.", PermissionLevel.WORKSPACE_WRITE, DocxAppendRequest, ("path",), handler=_append))
    registry.register(Tool("docx_export_pdf", "Export a DOCX to PDF using LibreOffice (if installed).", PermissionLevel.WORKSPACE_WRITE, DocxExportPdfRequest, ("path", "output"), handler=_export))
