"""The coding agent's tool catalogue.

Every tool works relative to one workspace root and refuses paths outside it (paths are
canonicalised, so ``..`` and symlink/junction escapes are blocked by the same rules as
the rest of the app). Each tool has a *kind* used by the approval policy:

* ``read``    — read/list/search/git-inspect: never needs approval
* ``edit``    — file writes/edits: previewed as a diff; approval depends on the mode
* ``command`` — shell commands and git commits: approval unless allow-listed
* ``network`` — web search/fetch: only offered when internet access is enabled
* ``meta``    — the todo list (no side effects outside the session)

Edit tools are split into ``prepare`` (compute the new content + unified diff, no side
effects) and ``apply`` (write it), so the user can approve exactly what will be written.
"""
from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from ...security.workspace import WorkspaceViolation, canonicalize
from ...tools.terminal.terminal import classify_command
from ...security.permissions import PermissionLevel
from .environment import IGNORED_DIRS, run_git

MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_OUTPUT_CHARS = 30_000
MAX_GREP_MATCHES = 200
MAX_GLOB_RESULTS = 500
MAX_FILE_BYTES = 2_000_000

# Extra destructive patterns (PowerShell / git) on top of terminal.DESTRUCTIVE_PATTERNS.
_EXTRA_DESTRUCTIVE = [
    r"\bremove-item\b[^|;]*-recurse",
    r"\brm\s+-r",
    r"\brmdir\b",
    r"\bdel\s+.*\*",
    r"\bgit\s+push\b[^|;]*(--force|-f\b)",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b[^|;]*-[a-z]*f",
    r"\bgit\s+checkout\s+--\s",
    r"\bformat-volume\b",
    r"\bclear-disk\b",
    r"\bset-executionpolicy\b",
    r"\bstop-process\b",
    r"\btaskkill\b",
    r"\binvoke-expression\b|\biex\b",
    r"\b(curl|iwr|invoke-webrequest)\b[^|;]*\|\s*(iex|sh|bash|python)",
]
_EXTRA_RE = [re.compile(p, re.IGNORECASE) for p in _EXTRA_DESTRUCTIVE]


def is_destructive(command: str) -> bool:
    if classify_command(command) == PermissionLevel.HIGH_IMPACT:
        return True
    return any(regex.search(command) for regex in _EXTRA_RE)


# ------------------------------------------------------------------ context
@dataclass
class Workspace:
    root: str
    internet_enabled: bool = False
    # Called with chunks of command output while a command runs (terminal view).
    on_output: Optional[Callable[[str], None]] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    todos: list[dict[str, Any]] = field(default_factory=list)
    search_fn: Optional[Callable[[str, int], list[dict[str, str]]]] = None
    fetch_fn: Optional[Callable[[str], dict[str, str]]] = None
    command_runner: Optional[Callable[..., dict[str, Any]]] = None

    def resolve(self, path: str) -> Path:
        if not path or not str(path).strip():
            raise WorkspaceViolation("A path is required.")
        raw = Path(str(path).strip().strip('"'))
        candidate = raw if raw.is_absolute() else Path(self.root) / raw
        target = canonicalize(candidate)
        root = canonicalize(self.root)
        key = os.path.normcase(str(target))
        root_key = os.path.normcase(str(root))
        if key != root_key and not key.startswith(root_key + os.sep):
            raise WorkspaceViolation(f"'{path}' is outside the workspace ({root}). Only files inside it can be used.")
        return target

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(canonicalize(self.root))).replace("\\", "/")
        except ValueError:
            return str(path)


@dataclass
class ToolOutcome:
    status: str  # success | error | denied | cancelled | timeout
    output: str  # text returned to the model
    data: dict[str, Any] = field(default_factory=dict)  # for the UI (diff, exit code, ...)


@dataclass
class EditPlan:
    path: Path
    rel: str
    before: Optional[str]  # None = file does not exist yet
    after: str
    diff: str
    added: int
    removed: int
    note: Optional[str] = None


class ToolError(Exception):
    """A tool failed in a way the model can fix (bad args, no match, ...)."""


# ------------------------------------------------------------------ schemas
class ReadFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="First line to read (1-based).")
    limit: int = Field(default=MAX_READ_LINES, ge=1, le=MAX_READ_LINES, description="Maximum number of lines.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace root.")
    content: str = Field(description="The complete new file content.")


class EditFileArgs(BaseModel):
    path: str = Field(description="File path, relative to the workspace root.")
    old_string: str = Field(description="Exact text to replace (must match the file, including indentation).")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of requiring a unique match.")


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Folder, relative to the workspace root.")


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob such as '**/*.py' or 'src/*.ts'.")
    path: str = Field(default=".", description="Folder to search in.")


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression (Python syntax).")
    path: str = Field(default=".", description="File or folder to search.")
    glob: Optional[str] = Field(default=None, description="Only search files matching this glob, e.g. '*.py'.")
    ignore_case: bool = False


class RunCommandArgs(BaseModel):
    command: str = Field(min_length=1, description="PowerShell command line to run in the workspace root folder.")
    timeout: int = Field(default=120, ge=1, le=1800, description="Seconds before the command is stopped.")


class GitDiffArgs(BaseModel):
    path: Optional[str] = Field(default=None, description="Limit the diff to this path.")
    staged: bool = Field(default=False, description="Show staged changes instead of unstaged ones.")


class GitLogArgs(BaseModel):
    count: int = Field(default=10, ge=1, le=100)


class GitCommitArgs(BaseModel):
    message: str = Field(min_length=1, description="Commit message.")
    paths: Optional[list[str]] = Field(default=None, description="Files to stage; omit to stage all changes.")


class NoArgs(BaseModel):
    pass


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)


class WebFetchArgs(BaseModel):
    url: str = Field(description="http(s) URL to fetch as text.")


class TodoItem(BaseModel):
    content: str = Field(min_length=1)
    status: Literal["pending", "in_progress", "completed"] = "pending"


class TodoArgs(BaseModel):
    todos: list[TodoItem] = Field(description="The complete, updated todo list (replaces the previous one).")


# ------------------------------------------------------------------ helpers
def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB; read a smaller file.")
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolError(f"{path.name} looks like a binary file.")
    text = data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep CRLF files CRLF on Windows.
    newline = "\r\n" if path.exists() and b"\r\n" in path.read_bytes()[:65536] else "\n"
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def unified_diff(rel: str, before: Optional[str], after: str) -> tuple[str, int, int]:
    a = (before or "").splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    lines = list(
        difflib.unified_diff(a, b, fromfile=f"a/{rel}" if before is not None else "/dev/null", tofile=f"b/{rel}", n=3)
    )
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    text = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines)
    return text, added, removed


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return f"{head}\n… [{len(text) - limit} characters omitted] …\n{tail}"


# --------------------------------------------------------------- read tools
def read_file(ws: Workspace, args: ReadFileArgs) -> ToolOutcome:
    path = ws.resolve(args.path)
    if not path.is_file():
        raise ToolError(f"File not found: {args.path}" + (" (it is a folder; use list_dir)" if path.is_dir() else ""))
    lines = _read_text(path).split("\n")
    start = args.offset - 1
    chunk = lines[start:start + args.limit]
    numbered = [f"{start + i + 1:>5}\t{line[:MAX_LINE_CHARS]}" for i, line in enumerate(chunk)]
    footer = ""
    if start + args.limit < len(lines):
        footer = f"\n… {len(lines) - start - args.limit} more lines (use offset={start + args.limit + 1})"
    text = "\n".join(numbered) + footer if chunk else "(empty file)"
    return ToolOutcome("success", text, {"path": ws.rel(path), "lines": len(lines)})


def list_dir(ws: Workspace, args: ListDirArgs) -> ToolOutcome:
    path = ws.resolve(args.path)
    if not path.is_dir():
        raise ToolError(f"Not a folder: {args.path}")
    entries = []
    for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if entry.name in IGNORED_DIRS:
            continue
        if entry.is_dir():
            entries.append(f"{entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{entry.name}  ({size} bytes)")
    return ToolOutcome("success", "\n".join(entries) or "(empty folder)", {"path": ws.rel(path), "count": len(entries)})


def glob_files(ws: Workspace, args: GlobArgs) -> ToolOutcome:
    base = ws.resolve(args.path)
    pattern = args.pattern.replace("\\", "/")
    results = []
    for file in _walk(base):
        rel = file.relative_to(base).as_posix()
        if fnmatch.fnmatch(rel, pattern) or ("/" not in pattern and fnmatch.fnmatch(file.name, pattern)) or (
            pattern.startswith("**/") and fnmatch.fnmatch(rel, pattern[3:])
        ):
            results.append(ws.rel(file))
            if len(results) >= MAX_GLOB_RESULTS:
                break
    results.sort()
    text = "\n".join(results) if results else "No files matched."
    return ToolOutcome("success", text, {"count": len(results)})


def grep(ws: Workspace, args: GrepArgs) -> ToolOutcome:
    try:
        regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc
    base = ws.resolve(args.path)
    files = [base] if base.is_file() else list(_walk(base))
    matches = []
    for file in files:
        if args.glob and not fnmatch.fnmatch(file.name, args.glob):
            continue
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            text = _read_text(file)
        except (ToolError, OSError):
            continue
        for number, line in enumerate(text.split("\n"), start=1):
            if regex.search(line):
                matches.append(f"{ws.rel(file)}:{number}: {line.strip()[:300]}")
                if len(matches) >= MAX_GREP_MATCHES:
                    break
        if len(matches) >= MAX_GREP_MATCHES:
            break
    text = "\n".join(matches) if matches else "No matches."
    if len(matches) >= MAX_GREP_MATCHES:
        text += f"\n… stopped after {MAX_GREP_MATCHES} matches; narrow the search."
    return ToolOutcome("success", text, {"count": len(matches)})


# --------------------------------------------------------------- edit tools
def prepare_write(ws: Workspace, args: WriteFileArgs) -> EditPlan:
    path = ws.resolve(args.path)
    if path.is_dir():
        raise ToolError(f"{args.path} is a folder.")
    before = _read_text(path) if path.is_file() else None
    after = args.content.replace("\r\n", "\n")
    if before == after:
        raise ToolError(f"{args.path} already has exactly this content; nothing to write.")
    rel = ws.rel(path)
    diff, added, removed = unified_diff(rel, before, after)
    return EditPlan(path, rel, before, after, diff, added, removed)


def prepare_edit(ws: Workspace, args: EditFileArgs) -> EditPlan:
    path = ws.resolve(args.path)
    old = args.old_string.replace("\r\n", "\n")
    new = args.new_string.replace("\r\n", "\n")
    if not old and new and (not path.exists() or (path.is_file() and not _read_text(path).strip())):
        # Small models use edit_file with an empty old_string to fill a new or empty file
        # (seen repeatedly on this machine). That intent is unambiguous: write the content.
        return prepare_write(ws, WriteFileArgs(path=args.path, content=new))
    if not path.is_file():
        raise ToolError(f"File not found: {args.path}. Use write_file to create a new file.")
    before = _read_text(path)
    if old == new:
        raise ToolError("old_string and new_string are identical; nothing to change.")
    if not old:
        raise ToolError(
            "old_string is empty. To insert text, set old_string to an existing line next to where the new text goes "
            "and repeat that line in new_string together with the new text. To replace the whole file, use write_file."
        )
    count = before.count(old)
    if count == 0:
        fuzzy = _whitespace_tolerant_replace(before, old, new)
        if fuzzy is not None:
            rel = ws.rel(path)
            diff, added, removed = unified_diff(rel, before, fuzzy)
            return EditPlan(path, rel, before, fuzzy, diff, added, removed, note="matched ignoring whitespace differences")
        hint = ""
        first = old.strip().splitlines()[0] if old.strip() else ""
        if first and first.strip() in before:
            hint = " The first line exists, so check indentation/whitespace of the following lines."
        raise ToolError(f"old_string was not found in {args.path}.{hint} Read the file again and copy the text exactly.")
    if count > 1 and not args.replace_all:
        raise ToolError(
            f"old_string occurs {count} times in {args.path}. Include more surrounding lines to make it unique, "
            "or set replace_all=true."
        )
    after = before.replace(old, new) if args.replace_all else before.replace(old, new, 1)
    rel = ws.rel(path)
    diff, added, removed = unified_diff(rel, before, after)
    return EditPlan(path, rel, before, after, diff, added, removed)


def _trim_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _whitespace_tolerant_replace(text: str, old: str, new: str) -> Optional[str]:
    """Fallback for edits whose old_string differs only in whitespace.

    Small models often add or drop blank lines around old_string, leave trailing spaces,
    or shift the indentation of the whole block. Lines are compared after stripping
    trailing whitespace (pass 1) or all surrounding whitespace (pass 2, then new_string
    is re-indented by the same offset). Applies only to a *unique* match.
    """
    old_lines = _trim_blank(old.split("\n"))
    new_lines = _trim_blank(new.split("\n"))
    if not old_lines:
        return None
    lines = text.split("\n")
    n = len(old_lines)
    for strict in (True, False):
        key = (lambda s: s.rstrip()) if strict else (lambda s: s.strip())
        target = [key(x) for x in old_lines]
        hits = [i for i in range(len(lines) - n + 1) if [key(x) for x in lines[i:i + n]] == target]
        if len(hits) > 1:
            return None
        if len(hits) == 1:
            i = hits[0]
            replacement = new_lines
            if not strict:
                file_indent = _indent(lines[i])
                old_indent = _indent(old_lines[0])
                replacement = []
                for line in new_lines:
                    if not line.strip():
                        replacement.append(line)
                    elif line.startswith(old_indent):
                        replacement.append(file_indent + line[len(old_indent):])
                    else:
                        replacement.append(file_indent + line.lstrip())
            return "\n".join(lines[:i] + replacement + lines[i + n:])
    return None


def changed_region(before: str, after: str, context: int = 3, max_lines: int = 60) -> str:
    """The edited lines of ``after`` with line numbers, so the model sees the result."""
    a, b = before.split("\n"), after.split("\n")
    if a and a[-1] == "":
        a = a[:-1]
    if b and b[-1] == "":
        b = b[:-1]
    changed = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not changed:
        return ""
    start = max(0, changed[0][3] - context)
    end = min(len(b), changed[-1][4] + context)
    if end - start > max_lines:
        end = start + max_lines
    numbered = "\n".join(f"{i + 1:>5}\t{b[i]}" for i in range(start, end))
    return f"Lines {start + 1}-{end} now read:\n{numbered}"


def syntax_problem(rel: str, text: str) -> Optional[str]:
    """Immediate diagnostics after an edit (Python and JSON), like an editor's red squiggle."""
    lower = rel.lower()
    try:
        if lower.endswith(".py"):
            compile(text, rel, "exec", dont_inherit=True)
        elif lower.endswith(".json"):
            import json

            json.loads(text)
    except SyntaxError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except ValueError as exc:
        return str(exc)
    return None


def apply_plan(plan: EditPlan) -> ToolOutcome:
    current = _read_text(plan.path) if plan.path.is_file() else None
    if current != plan.before:
        raise ToolError(f"{plan.rel} changed on disk after the edit was prepared. Read it again and redo the edit.")
    _write_text(plan.path, plan.after)
    verb = "Created" if plan.before is None else "Updated"
    message = f"{verb} {plan.rel} (+{plan.added} -{plan.removed} lines)."
    if plan.note:
        message += f" Note: {plan.note}."
    if plan.before is not None:
        message += "\n" + changed_region(plan.before, plan.after)
    problem = syntax_problem(plan.rel, plan.after)
    if problem:
        message += f"\nWARNING: {plan.rel} now has a syntax error: {problem}. Fix it before doing anything else."
    return ToolOutcome(
        "success",
        message,
        {"path": plan.rel, "diff": plan.diff, "added": plan.added, "removed": plan.removed, "created": plan.before is None},
    )


# ------------------------------------------------------------- command tool
def _powershell_command(command: str) -> list[str]:
    # UTF-8 output so Python/pytest output is not mangled by the OEM code page.
    prefix = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8; "
    return ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", prefix + command]


def _shell_command(command: str) -> list[str]:
    if sys.platform == "win32":
        return _powershell_command(command)
    return ["/bin/bash", "-lc", command]


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def default_command_runner(
    argv: list[str], cwd: str, timeout: int, on_output: Optional[Callable[[str], None]], cancel: threading.Event
) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.setdefault("NO_COLOR", "1")
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    started = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        **kwargs,
    )
    chunks: list[str] = []

    def pump() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
            chunks.append(text)
            if on_output:
                try:
                    on_output(text)
                except Exception:  # noqa: BLE001
                    pass

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = cancelled = False
    while proc.poll() is None:
        if cancel.is_set():
            cancelled = True
            _kill_tree(proc)
            break
        if time.monotonic() - started > timeout:
            timed_out = True
            _kill_tree(proc)
            break
        time.sleep(0.05)
    proc.wait(timeout=10)
    reader.join(timeout=5)
    return {
        "exit_code": proc.returncode,
        "output": "".join(chunks),
        "elapsed": time.monotonic() - started,
        "timed_out": timed_out,
        "cancelled": cancelled,
    }


def run_command(ws: Workspace, args: RunCommandArgs) -> ToolOutcome:
    # Always the workspace root: small models mis-set a working folder and then use
    # root-relative paths in the command (seen on this machine; DECISIONS D5).
    cwd = ws.resolve(".")
    runner = ws.command_runner or default_command_runner
    result = runner(_shell_command(args.command), str(cwd), args.timeout, ws.on_output, ws.cancel)
    output = _truncate(result.get("output", "").strip())
    data = {
        "command": args.command,
        "cwd": ".",
        "exit_code": result.get("exit_code"),
        "elapsed": round(result.get("elapsed") or 0, 2),
        "output": output,
    }
    if result.get("cancelled"):
        return ToolOutcome("cancelled", f"Command was stopped by the user.\n{output}", data)
    if result.get("timed_out"):
        return ToolOutcome("timeout", f"Command timed out after {args.timeout}s and was stopped.\n{output}", data)
    code = result.get("exit_code")
    header = f"Exit code {code} ({data['elapsed']}s)."
    return ToolOutcome("success" if code == 0 else "error", f"{header}\n{output}" if output else header, data)


# ---------------------------------------------------------------- git tools
def _git(ws: Workspace, args: list[str]) -> str:
    out = run_git(ws.root, args, timeout=30)
    if out is None:
        if run_git(ws.root, ["rev-parse", "--is-inside-work-tree"]) is None:
            raise ToolError("The workspace is not a git repository (run `git init` first if you need git).")
        raise ToolError(f"git {' '.join(args)} failed.")
    return out


def git_status(ws: Workspace, args: NoArgs) -> ToolOutcome:
    out = _git(ws, ["status", "--short", "--branch"])
    return ToolOutcome("success", out.strip() or "Working tree clean.")


def git_diff(ws: Workspace, args: GitDiffArgs) -> ToolOutcome:
    cmd = ["diff", "--no-color"] + (["--staged"] if args.staged else [])
    if args.path:
        cmd += ["--", str(ws.resolve(args.path))]
    out = _git(ws, cmd)
    return ToolOutcome("success", _truncate(out.strip()) or "No changes.", {"diff": _truncate(out)})


def git_log(ws: Workspace, args: GitLogArgs) -> ToolOutcome:
    out = _git(ws, ["log", f"-{args.count}", "--pretty=format:%h %ad %an  %s", "--date=short"])
    return ToolOutcome("success", out.strip() or "No commits yet.")


def git_commit(ws: Workspace, args: GitCommitArgs) -> ToolOutcome:
    if args.paths:
        for p in args.paths:
            ws.resolve(p)
        _git(ws, ["add", "--", *args.paths])
    else:
        _git(ws, ["add", "-A"])
    out = run_git(ws.root, ["commit", "-m", args.message], timeout=60)
    if out is None:
        status = run_git(ws.root, ["status", "--short"]) or ""
        raise ToolError("git commit failed" + (" (nothing to commit)." if not status.strip() else ". Check git config user.name/email."))
    return ToolOutcome("success", out.strip())


# ---------------------------------------------------------------- web tools
def web_search(ws: Workspace, args: WebSearchArgs) -> ToolOutcome:
    if not ws.internet_enabled:
        raise ToolError("Internet access is disabled in Settings.")
    from ...tools.web.web import _default_search

    results = (ws.search_fn or _default_search)(args.query, args.max_results)
    text = "\n\n".join(f"{r.get('title')}\n{r.get('url')}\n{r.get('snippet', '')}" for r in results) or "No results."
    return ToolOutcome("success", text, {"results": results})


def web_fetch(ws: Workspace, args: WebFetchArgs) -> ToolOutcome:
    if not ws.internet_enabled:
        raise ToolError("Internet access is disabled in Settings.")
    if not re.match(r"^https?://", args.url):
        raise ToolError("Only http(s) URLs can be fetched.")
    from ...tools.web.web import _default_fetch

    page = (ws.fetch_fn or _default_fetch)(args.url)
    return ToolOutcome("success", _truncate(page.get("text", ""), 20_000), {"url": args.url, "title": page.get("title")})


# ---------------------------------------------------------------- todo tool
def todo_write(ws: Workspace, args: TodoArgs) -> ToolOutcome:
    ws.todos = [item.model_dump() for item in args.todos]
    lines = [f"[{'x' if t['status'] == 'completed' else ('~' if t['status'] == 'in_progress' else ' ')}] {t['content']}" for t in ws.todos]
    return ToolOutcome("success", "Todo list updated:\n" + "\n".join(lines), {"todos": ws.todos})


# ---------------------------------------------------------------- catalogue
@dataclass
class AgentTool:
    name: str
    description: str
    kind: str  # read | edit | command | network | meta
    args_model: type[BaseModel]
    run: Optional[Callable[[Workspace, Any], ToolOutcome]] = None
    prepare: Optional[Callable[[Workspace, Any], EditPlan]] = None

    def schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.schema()}}

    def validate(self, args: Any) -> BaseModel:
        if not isinstance(args, dict):
            raise ToolError(f"Arguments for {self.name} must be a JSON object.")
        try:
            return self.args_model.model_validate(args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            raise ToolError(f"Invalid arguments for {self.name}: {problems}") from exc


TOOLS: list[AgentTool] = [
    AgentTool("read_file", "Read a text file with line numbers. Use offset/limit for large files.", "read", ReadFileArgs, run=read_file),
    AgentTool("list_dir", "List the files and folders in a workspace folder.", "read", ListDirArgs, run=list_dir),
    AgentTool("glob", "Find files by name pattern, e.g. '**/*.py'.", "read", GlobArgs, run=glob_files),
    AgentTool("grep", "Search file contents with a regular expression; returns path:line: text.", "read", GrepArgs, run=grep),
    AgentTool("git_status", "Show the git branch and changed files.", "read", NoArgs, run=git_status),
    AgentTool("git_diff", "Show uncommitted changes (unified diff).", "read", GitDiffArgs, run=git_diff),
    AgentTool("git_log", "Show recent commits.", "read", GitLogArgs, run=git_log),
    AgentTool("todo_write", "Create or update your task list. Send the full list each time with statuses.", "meta", TodoArgs, run=todo_write),
    AgentTool("write_file", "Create a new file or completely replace a file's content.", "edit", WriteFileArgs, prepare=prepare_write),
    AgentTool(
        "edit_file",
        "Replace an exact string in a file (old_string must be unique unless replace_all). Read the file first.",
        "edit",
        EditFileArgs,
        prepare=prepare_edit,
    ),
    AgentTool(
        "run_command",
        "Run a PowerShell command in the workspace (e.g. tests, builds, scripts). Output is returned when it exits.",
        "command",
        RunCommandArgs,
        run=run_command,
    ),
    AgentTool("git_commit", "Stage files (all by default) and create a git commit.", "command", GitCommitArgs, run=git_commit),
    AgentTool("web_search", "Search the web (only when internet access is enabled).", "network", WebSearchArgs, run=web_search),
    AgentTool("web_fetch", "Fetch a web page as text (only when internet access is enabled).", "network", WebFetchArgs, run=web_fetch),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Tools available per mode. Ask/Plan never see edit or command tools.
_READ_ONLY = {"read", "meta", "network"}


def tools_for_mode(mode: str, internet_enabled: bool) -> list[AgentTool]:
    if mode == "chat":
        # Plain chat has no workspace: only web lookups (when internet access is on).
        allowed = {"network"}
    elif mode in ("ask", "plan"):
        allowed = _READ_ONLY
    else:
        allowed = {"read", "meta", "network", "edit", "command"}
    return [t for t in TOOLS if t.kind in allowed and (t.kind != "network" or internet_enabled)]
