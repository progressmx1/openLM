"""Text tool-call protocol and streaming tag splitter.

Models without native tool calling are asked to emit::

    <tool_call>
    {"name": "read_file", "arguments": {"path": "a.py"}}
    </tool_call>

The same parser also rescues tool calls that a native-capable model wrote into plain
content (llama.cpp only extracts calls it can parse). ``<think>…</think>`` blocks are
separated from visible text while streaming, even when a tag is split across chunks.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

TAGS = {
    "<think>": ("reasoning", True),
    "</think>": ("reasoning", False),
    "<tool_call>": ("tool", True),
    "</tool_call>": ("tool", False),
}
_MAX_TAG = max(len(t) for t in TAGS)


class StreamSplitter:
    """Incrementally split model text into ``content`` / ``reasoning`` / ``tool`` parts.

    ``feed()`` returns a list of ``(kind, text)``; ``kind`` is ``"content"``,
    ``"reasoning"``, or ``"tool_block"`` (a complete ``<tool_call>`` body).
    """

    def __init__(self, start_in_reasoning: bool = False) -> None:
        self.buffer = ""
        self.state = "reasoning" if start_in_reasoning else "content"
        self.tool_text = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buffer += text
        out: list[tuple[str, str]] = []
        while True:
            lower = self.buffer.lower()
            positions = [(lower.find(tag), tag) for tag in TAGS if lower.find(tag) != -1]
            if not positions:
                # Keep a possible partial tag at the end of the buffer.
                keep = 0
                for size in range(1, _MAX_TAG):
                    if any(tag.startswith(lower[-size:]) for tag in TAGS) and size <= len(lower):
                        keep = size
                emit = self.buffer[: len(self.buffer) - keep] if keep else self.buffer
                self.buffer = self.buffer[len(emit):]
                self._emit(out, emit)
                return out
            index, tag = min(positions)
            self._emit(out, self.buffer[:index])
            self.buffer = self.buffer[index + len(tag):]
            kind, opening = TAGS[tag]
            if kind == "reasoning":
                if opening and self.state == "content":
                    self.state = "reasoning"
                elif not opening and self.state == "reasoning":
                    self.state = "content"
            else:
                if opening and self.state == "content":
                    self.state = "tool"
                    self.tool_text = ""
                elif not opening and self.state == "tool":
                    out.append(("tool_block", self.tool_text))
                    self.tool_text = ""
                    self.state = "content"

    def _emit(self, out: list[tuple[str, str]], text: str) -> None:
        if not text:
            return
        if self.state == "tool":
            self.tool_text += text
        else:
            out.append((self.state, text))

    def close(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        rest, self.buffer = self.buffer, ""
        self._emit(out, rest)
        if self.state == "tool" and self.tool_text.strip():
            # Unterminated block at end of output: still try to use it.
            out.append(("tool_block", self.tool_text))
            self.tool_text = ""
        self.state = "content"
        return out


def _loads_lenient(text: str) -> Optional[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Trailing commas, single trailing garbage.
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    start, end = fixed.find("{"), fixed.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(fixed[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_tool_block(text: str) -> dict[str, Any]:
    """Parse one ``<tool_call>`` body. Returns ``{"name", "arguments"}`` or ``{"error"}``."""
    data = _loads_lenient(text)
    if not isinstance(data, dict):
        return {"error": "The tool call is not valid JSON. Use {\"name\": ..., \"arguments\": {...}}.", "raw": text}
    name = data.get("name") or data.get("tool") or data.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        data, name = name, name.get("name")
    args = data.get("arguments", data.get("args", data.get("parameters", {})))
    if isinstance(args, str):
        parsed = _loads_lenient(args)
        args = parsed if isinstance(parsed, dict) else {}
    if not name or not isinstance(name, str):
        return {"error": "The tool call has no \"name\".", "raw": text}
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


_FENCED = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.DOTALL)


def find_fenced_tool_calls(text: str, known: set[str]) -> list[dict[str, Any]]:
    """Fallback: a JSON code block that names a known tool (weak models often do this)."""
    calls = []
    for match in _FENCED.finditer(text):
        parsed = parse_tool_block(match.group(1))
        if parsed.get("name") in known:
            calls.append(parsed)
    return calls


_FUNCTION_ATTR = re.compile(
    r"<function\s+name\s*=\s*[\"'](?P<name>\w+)[\"']\s+(?:arguments|args|parameters)\s*=\s*(?P<q>[\"'])(?P<args>.*?)(?P=q)\s*/?>",
    re.DOTALL,
)
_FUNCTION_EQ = re.compile(r"<function=(?P<name>\w+)>(?P<body>.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)


def rescue_tool_calls(text: str, known: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Recover tool calls that weak models write in improvised formats.

    Seen on this machine with Qwen2.5-Coder-7B (whose template asks for ``<tool_call>``):
    ``<function name="write_file" arguments='{...}'/>`` and ``<{"name": ..., "arguments": ...}>``.
    Also handles Qwen3-Coder style ``<function=name><parameter=k>v</parameter></function>``
    and any bare JSON object ``{"name": <known tool>, "arguments": {...}}``.
    Returns (calls, remaining_text).
    """
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for match in _FUNCTION_ATTR.finditer(text):
        if match.group("name") in known:
            args = _loads_lenient(match.group("args"))
            calls.append({"name": match.group("name"), "arguments": args if isinstance(args, dict) else {}})
            spans.append(match.span())
    for match in _FUNCTION_EQ.finditer(text):
        if match.group("name") in known:
            body = match.group("body")
            params = {m.group("key"): m.group("value") for m in _PARAMETER.finditer(body)}
            if not params:
                parsed = _loads_lenient(body)
                params = parsed if isinstance(parsed, dict) else {}
            calls.append({"name": match.group("name"), "arguments": params})
            spans.append(match.span())
    if not calls:
        decoder = json.JSONDecoder()
        index = 0
        while True:
            start = text.find("{", index)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(obj, dict):
                parsed = parse_tool_block(json.dumps(obj))
                if parsed.get("name") in known:
                    calls.append(parsed)
                    spans.append((start, end))
            index = end
    remaining = text
    for start, end in sorted(spans, reverse=True):
        remaining = remaining[:start] + remaining[end:]
    # Drop now-empty code fences and stray wrapper characters left around the calls.
    remaining = re.sub(r"```[a-zA-Z]*\s*[<>]*\s*```", "", remaining)
    remaining = re.sub(r"(?m)^\s*[<>]+\s*$", "", remaining)
    return calls, remaining.strip()


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():  # opening tag was in the prompt (DeepSeek-R1 template)
        text = text[text.lower().rfind("</think>") + len("</think>"):]
    return text.strip()
