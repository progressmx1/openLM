"""PDF operations built on PyMuPDF.

Pure functions operate on plain paths (used by the attachment/extract endpoint,
which is user-initiated and therefore not workspace-restricted). The ``Tool``
wrappers re-check workspace boundaries so the agent/framework path stays secure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pymupdf
from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel

DEFAULT_FONT_SIZE = 11


def pdf_metadata(path: str | Path) -> dict[str, Any]:
    doc = pymupdf.open(str(path))
    try:
        meta = doc.metadata or {}
        return {
            "path": str(Path(path).resolve()),
            "page_count": doc.page_count,
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "keywords": meta.get("keywords"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "format": meta.get("format"),
        }
    finally:
        doc.close()


def pdf_extract_text(path: str | Path, pages: Optional[list[int]] = None) -> dict[str, Any]:
    doc = pymupdf.open(str(path))
    try:
        selected = pages or list(range(1, doc.page_count + 1))
        result = []
        for page_no in selected:
            if page_no < 1 or page_no > doc.page_count:
                continue
            page = doc.load_page(page_no - 1)
            result.append({"page": page_no, "text": page.get_text()})
        return {"page_count": doc.page_count, "pages": result}
    finally:
        doc.close()


def pdf_search(
    path: str | Path, query: str, max_results: int = 10
) -> dict[str, Any]:
    doc = pymupdf.open(str(path))
    try:
        matches = []
        for page_no in range(1, doc.page_count + 1):
            if len(matches) >= max_results:
                break
            page = doc.load_page(page_no - 1)
            rects = page.search_for(query)
            if not rects:
                continue
            text = page.get_text()
            for rect in rects[: max_results - len(matches)]:
                # Grab a short excerpt around the match.
                words = page.get_text("words")
                snippet = _excerpt_around(words, rect)
                matches.append(
                    {
                        "page": page_no,
                        "bbox": [round(x, 1) for x in rect],
                        "excerpt": snippet,
                    }
                )
                if len(matches) >= max_results:
                    break
        return {"query": query, "count": len(matches), "matches": matches}
    finally:
        doc.close()


def _excerpt_around(words: list, rect) -> str:
    # words are [x0, y0, x1, y1, word, block, line, word_no]
    center_y = (rect.y0 + rect.y1) / 2
    near = [w for w in words if abs((w[1] + w[3]) / 2 - center_y) < 30]
    near.sort(key=lambda w: (round(w[7] / 100), w[0]))
    return " ".join(w[4] for w in near[:60])


def pdf_create(
    path: str | Path, content: str, title: Optional[str] = None
) -> dict[str, Any]:
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        if title:
            page.insert_text((72, 72), title, fontsize=16, fontname="helv")
            start_y = 100
        else:
            start_y = 72
        rect = pymupdf.Rect(72, start_y, page.rect.width - 72, page.rect.height - 72)
        # insert_textbox auto-wraps and returns unused height.
        page.insert_textbox(rect, content, fontsize=DEFAULT_FONT_SIZE, fontname="helv")
        doc.set_metadata({"title": title} if title else {})
        doc.save(str(path))
    finally:
        doc.close()
    return {"path": str(Path(path).resolve()), "page_count": 1}


def pdf_merge(paths: list[str], output: str) -> dict[str, Any]:
    merged = pymupdf.open()
    try:
        for path in paths:
            src = pymupdf.open(path)
            merged.insert_pdf(src)
            src.close()
        merged.save(output)
    finally:
        merged.close()
    return {"path": str(Path(output).resolve()), "page_count": pymupdf.open(output).page_count}


def pdf_extract_pages(path: str, pages: list[int], output: str) -> dict[str, Any]:
    src = pymupdf.open(path)
    out = pymupdf.open()
    try:
        for page_no in pages:
            if 1 <= page_no <= src.page_count:
                out.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
        out.save(output)
    finally:
        out.close()
        src.close()
    return {"path": str(Path(output).resolve()), "page_count": pymupdf.open(output).page_count}


def pdf_rotate(
    path: str, pages: Optional[list[int]], degrees: int, output: str
) -> dict[str, Any]:
    src = pymupdf.open(path)
    try:
        selected = pages or list(range(1, src.page_count + 1))
        for page_no in selected:
            if 1 <= page_no <= src.page_count:
                page = src.load_page(page_no - 1)
                page.set_rotation((page.rotation + degrees) % 360)
        src.save(output)
    finally:
        src.close()
    return {"path": str(Path(output).resolve())}


def pdf_convert(path: str, fmt: str, output: str) -> dict[str, Any]:
    extracted = pdf_extract_text(path)
    text = "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in extracted["pages"])
    if fmt == "txt":
        Path(output).write_text(text, encoding="utf-8")
    elif fmt == "docx":
        from .docx import docx_create_from_text

        docx_create_from_text(output, text)
    else:
        raise ValueError(f"Unsupported conversion format: {fmt}")
    return {"path": str(Path(output).resolve())}


def pdf_detect_scanned(path: str) -> dict[str, Any]:
    doc = pymupdf.open(str(path))
    try:
        scanned_pages = []
        for page_no in range(1, doc.page_count + 1):
            page = doc.load_page(page_no - 1)
            has_text = bool(page.get_text().strip())
            has_images = bool(page.get_images())
            if has_images and not has_text:
                scanned_pages.append(page_no)
        return {
            "page_count": doc.page_count,
            "image_only_pages": scanned_pages,
            "needs_ocr": bool(scanned_pages),
        }
    finally:
        doc.close()


def ocr_pdf(path: str, output: Optional[str] = None) -> dict[str, Any]:
    """OCR an image-only PDF using Tesseract (reports clearly when it is missing)."""
    import shutil
    import subprocess
    import tempfile

    tesseract = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if not tesseract:
        raise RuntimeError(
            "OCR requires Tesseract, which is not installed or not on PATH. "
            "Install it and re-run, or extract the images manually."
        )

    doc = pymupdf.open(str(path))
    texts: list[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for page_no in range(doc.page_count):
                page = doc.load_page(page_no)
                pix = page.get_pixmap(dpi=200)
                img_path = Path(tmp) / f"page-{page_no + 1}.png"
                pix.save(str(img_path))
                proc = subprocess.run(
                    [tesseract, str(img_path), "stdout"],
                    capture_output=True, text=True, errors="replace", timeout=120,
                )
                texts.append(proc.stdout)
    finally:
        doc.close()

    text = "\n\n".join(t for t in texts if t.strip())
    if output:
        Path(output).write_text(text, encoding="utf-8")
    return {"ocr": True, "text": text, "output": str(Path(output).resolve()) if output else None}


# ------------------------------------------------------------------ tools

class PdfReadRequest(BaseModel):
    path: str


class PdfExtractRequest(BaseModel):
    path: str
    pages: Optional[list[int]] = None


class PdfSearchRequest(BaseModel):
    path: str
    query: str
    max_results: int = Field(default=10, ge=1, le=100)


class PdfCreateRequest(BaseModel):
    path: str
    content: str
    title: Optional[str] = None


class PdfMergeRequest(BaseModel):
    paths: list[str]
    output: str


class PdfExtractPagesRequest(BaseModel):
    path: str
    pages: list[int]
    output: str


class PdfRotateRequest(BaseModel):
    path: str
    pages: Optional[list[int]] = None
    degrees: int
    output: str


class PdfConvertRequest(BaseModel):
    path: str
    format: str
    output: str


class PdfDetectScannedRequest(BaseModel):
    path: str


class PdfOcrRequest(BaseModel):
    path: str
    output: Optional[str] = None


def _resolve(ctx: ExecutionContext, path: str) -> str:
    return ctx.resolve_path(path)


def _err(exc: Exception) -> ToolResult:
    return ToolResult("error", error=str(exc))


def register_pdf_tools(registry: ToolRegistry) -> None:
    def _read(ctx: ExecutionContext, args: PdfReadRequest) -> ToolResult:
        try:
            return ToolResult("success", data=pdf_metadata(_resolve(ctx, args.path)))
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _extract(ctx: ExecutionContext, args: PdfExtractRequest) -> ToolResult:
        try:
            return ToolResult("success", data=pdf_extract_text(_resolve(ctx, args.path), args.pages))
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _search(ctx: ExecutionContext, args: PdfSearchRequest) -> ToolResult:
        try:
            return ToolResult(
                "success", data=pdf_search(_resolve(ctx, args.path), args.query, args.max_results)
            )
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _create(ctx: ExecutionContext, args: PdfCreateRequest) -> ToolResult:
        try:
            return ToolResult(
                "success", data=pdf_create(_resolve(ctx, args.path), args.content, args.title)
            )
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _merge(ctx: ExecutionContext, args: PdfMergeRequest) -> ToolResult:
        try:
            paths = [_resolve(ctx, p) for p in args.paths]
            return ToolResult("success", data=pdf_merge(paths, _resolve(ctx, args.output)))
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _extract_pages(ctx: ExecutionContext, args: PdfExtractPagesRequest) -> ToolResult:
        try:
            return ToolResult(
                "success",
                data=pdf_extract_pages(_resolve(ctx, args.path), args.pages, _resolve(ctx, args.output)),
            )
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _rotate(ctx: ExecutionContext, args: PdfRotateRequest) -> ToolResult:
        try:
            return ToolResult(
                "success",
                data=pdf_rotate(_resolve(ctx, args.path), args.pages, args.degrees, _resolve(ctx, args.output)),
            )
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _convert(ctx: ExecutionContext, args: PdfConvertRequest) -> ToolResult:
        try:
            return ToolResult(
                "success",
                data=pdf_convert(_resolve(ctx, args.path), args.format, _resolve(ctx, args.output)),
            )
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _detect_scanned(ctx: ExecutionContext, args: PdfDetectScannedRequest) -> ToolResult:
        try:
            return ToolResult("success", data=pdf_detect_scanned(_resolve(ctx, args.path)))
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    def _ocr(ctx: ExecutionContext, args: PdfOcrRequest) -> ToolResult:
        try:
            output = _resolve(ctx, args.output) if args.output else None
            return ToolResult("success", data=ocr_pdf(_resolve(ctx, args.path), output))
        except Exception as exc:  # noqa: BLE001
            return _err(exc)

    registry.register(Tool("pdf_read", "Read PDF metadata and page count.", PermissionLevel.READ_ONLY, PdfReadRequest, ("path",), handler=_read))
    registry.register(Tool("pdf_extract_text", "Extract PDF text page by page.", PermissionLevel.READ_ONLY, PdfExtractRequest, ("path",), handler=_extract))
    registry.register(Tool("pdf_search", "Search a PDF for a phrase and return page numbers and excerpts.", PermissionLevel.READ_ONLY, PdfSearchRequest, ("path",), handler=_search))
    registry.register(Tool("pdf_create", "Create a PDF from text.", PermissionLevel.WORKSPACE_WRITE, PdfCreateRequest, ("path",), handler=_create))
    registry.register(Tool("pdf_merge", "Merge PDF files into one.", PermissionLevel.WORKSPACE_WRITE, PdfMergeRequest, ("paths", "output"), handler=_merge))
    registry.register(Tool("pdf_extract_pages", "Extract selected pages into a new PDF.", PermissionLevel.WORKSPACE_WRITE, PdfExtractPagesRequest, ("path", "output"), handler=_extract_pages))
    registry.register(Tool("pdf_rotate", "Rotate pages and save a new PDF.", PermissionLevel.WORKSPACE_WRITE, PdfRotateRequest, ("path", "output"), handler=_rotate))
    registry.register(Tool("pdf_convert", "Convert PDF text to TXT or DOCX.", PermissionLevel.WORKSPACE_WRITE, PdfConvertRequest, ("path", "output"), handler=_convert))
    registry.register(Tool("pdf_detect_scanned", "Detect image-only (scanned) PDF pages that may need OCR.", PermissionLevel.READ_ONLY, PdfDetectScannedRequest, ("path",), handler=_detect_scanned))
    registry.register(Tool("pdf_ocr", "OCR a scanned PDF using Tesseract (requires Tesseract installed).", PermissionLevel.WORKSPACE_WRITE, PdfOcrRequest, ("path", "output"), handler=_ocr))
