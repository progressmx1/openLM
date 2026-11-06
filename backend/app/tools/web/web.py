"""Opt-in internet tools: search, page fetch, and download.

All network access is gated by the ``internet_enabled`` setting and recorded in the
audit log. The search and fetch backends are injectable so tests never touch the
network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from lxml import html
from pydantic import BaseModel, Field

from ..base import ExecutionContext, Tool, ToolResult
from ..registry import ToolRegistry
from ...security.permissions import PermissionLevel

USER_AGENT = "LocalAIDesktop/0.1 (+local research tool)"

SEARCH_URL = "https://html.duckduckgo.com/html/"


def _has_class(name: str) -> str:
    """XPath predicate for an element carrying CSS class ``name`` (no cssselect needed)."""
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {name} ')"


def _real_url(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<url>; return the target URL."""
    from urllib.parse import parse_qs, unquote, urlparse

    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def parse_search_results(page: str, max_results: int) -> list[dict[str, str]]:
    tree = html.fromstring(page)
    results: list[dict[str, str]] = []
    for result in tree.xpath(f"//div[{_has_class('result')}]"):
        if len(results) >= max_results:
            break
        if result.xpath(f"self::*[{_has_class('result--ad')}]"):
            continue
        link_el = result.xpath(f".//a[{_has_class('result__a')}]")
        if not link_el:
            continue
        snippet_el = result.xpath(f".//*[{_has_class('result__snippet')}]")
        results.append({
            "title": link_el[0].text_content().strip(),
            "url": _real_url(link_el[0].get("href", "")),
            "snippet": snippet_el[0].text_content().strip() if snippet_el else "",
        })
    return results


def _default_search(query: str, max_results: int) -> list[dict[str, str]]:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as client:
        response = client.get(SEARCH_URL, params={"q": query})
        response.raise_for_status()
    return parse_search_results(response.text, max_results)


def page_text(content: bytes, url: str) -> dict[str, str]:
    tree = html.fromstring(content)
    title_el = tree.xpath("//title")
    title = title_el[0].text_content().strip() if title_el else url
    for bad in tree.xpath("//script | //style | //nav | //footer | //header | //aside | //noscript | //form | //svg"):
        bad.drop_tree()
    # Prefer the page's main content (skips site chrome such as GitHub's banners).
    candidates = tree.xpath("//main | //article | //*[@role='main']")
    root = max(candidates, key=lambda el: len(el.text_content())) if candidates else tree
    text = " ".join(" ".join(root.xpath(".//text()")).split())
    return {"title": title, "url": url, "text": text[:20_000]}


def _default_fetch(url: str) -> dict[str, str]:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return page_text(response.content, str(response.url))


def _default_download(url: str, output: str) -> dict[str, Any]:
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    Path(output).write_bytes(response.content)
    return {"path": str(Path(output).resolve()), "size_bytes": len(response.content)}


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=8, ge=1, le=20)


class WebFetchRequest(BaseModel):
    url: str = Field(min_length=1)


class WebDownloadRequest(BaseModel):
    url: str = Field(min_length=1)
    output: str


def _require_internet(ctx: ExecutionContext) -> Optional[ToolResult]:
    if not ctx.internet_enabled:
        return ToolResult("error", error="Internet access is disabled. Enable it in Settings.")
    return None


def make_web_tools(
    search: Optional[Callable[[str, int], list[dict[str, str]]]] = None,
    fetch: Optional[Callable[[str], dict[str, str]]] = None,
    download: Optional[Callable[[str, str], dict[str, Any]]] = None,
) -> list[Tool]:
    search_fn = search or _default_search
    fetch_fn = fetch or _default_fetch
    download_fn = download or _default_download

    def _search(ctx: ExecutionContext, args: WebSearchRequest) -> ToolResult:
        gate = _require_internet(ctx)
        if gate:
            return gate
        try:
            results = search_fn(args.query, args.max_results)
            return ToolResult("success", output=f"{len(results)} results", data={"results": results})
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _fetch(ctx: ExecutionContext, args: WebFetchRequest) -> ToolResult:
        gate = _require_internet(ctx)
        if gate:
            return gate
        try:
            data = fetch_fn(args.url)
            return ToolResult("success", output=data["text"], data=data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    def _download(ctx: ExecutionContext, args: WebDownloadRequest) -> ToolResult:
        gate = _require_internet(ctx)
        if gate:
            return gate
        try:
            data = download_fn(args.url, ctx.resolve_path(args.output))
            return ToolResult("success", output=f"Downloaded {data['size_bytes']} bytes", data=data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult("error", error=str(exc))

    return [
        Tool(
            "web_search",
            "Search the web using DuckDuckGo (requires internet access).",
            PermissionLevel.READ_ONLY,
            WebSearchRequest,
            handler=_search,
        ),
        Tool(
            "web_fetch",
            "Fetch and extract the readable text of a web page (requires internet access).",
            PermissionLevel.SYSTEM_ACTION,
            WebFetchRequest,
            handler=_fetch,
        ),
        Tool(
            "web_download",
            "Download a file from a URL into an approved workspace (requires internet access).",
            PermissionLevel.SYSTEM_ACTION,
            WebDownloadRequest,
            ("output",),
            handler=_download,
        ),
    ]


def register_web_tools(
    registry: ToolRegistry,
    search: Optional[Callable[[str, int], list[dict[str, str]]]] = None,
    fetch: Optional[Callable[[str], dict[str, str]]] = None,
    download: Optional[Callable[[str, str], dict[str, Any]]] = None,
) -> None:
    for tool in make_web_tools(search, fetch, download):
        registry.register(tool)
