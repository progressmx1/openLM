"""Coding-agent service: sessions, the multi-step agent loop, approvals and checkpoints.

One background thread runs each turn. Everything it does is published as events
(append-only; the UI merges ``tool``/``tool_update`` by ``call_id``):

=================  ===========================================================
``user``           the user's message
``delta``          streamed text: ``{"channel": "content"|"reasoning", "text"}``
                   (live only; the final text is persisted as ``assistant``)
``assistant``      a finished model message ``{"content", "reasoning"}``
``tool``           a tool call ``{"call_id", "name", "args", "kind", "status",
                   "diff"?, "risk"?}`` — status ``awaiting_approval`` asks the user
``tool_update``    ``{"call_id", "status", "output", "data"}``
``tool_output``    live command output ``{"call_id", "text"}`` (live only)
``todos``          the agent's todo list
``usage``          ``{"prompt_tokens", "completion_tokens", "context_used",
                   "context_max", "tokens_per_second"}``
``notice``         informational line (compaction, mode switch, fallback, …)
``error``          a user-facing error message
``turn_start`` / ``turn_end``  ``{"reason": "completed"|"interrupted"|"step_budget"|"stuck"|"error"}``
=================  ===========================================================

Safety rules: model output is never treated as approval — only :meth:`decide` (an HTTP
call from the UI) approves a pending action. Destructive commands always need approval,
even in auto mode or when allow-listed.
"""
from __future__ import annotations

import json
import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ...core.logging import get_logger
from ...core.settings import SettingsStore
from ...security.audit import AuditLog
from ...security.workspace import WorkspaceViolation, canonicalize
from .environment import environment_values
from .prompts import PromptLibrary
from .protocol import StreamSplitter, find_fenced_tool_calls, parse_tool_block, rescue_tool_calls, strip_think
from .store import AgentStore
from .tools import (
    TOOLS_BY_NAME,
    AgentTool,
    EditPlan,
    ToolError,
    ToolOutcome,
    Workspace,
    _write_text,
    apply_plan,
    is_destructive,
    tools_for_mode,
    unified_diff,
)

log = get_logger(__name__)

MODES = ("ask", "plan", "agent", "auto", "chat")
CODE_MODES = ("ask", "plan", "agent", "auto")
SUMMARY_PREFIX = "Summary of the earlier conversation (compacted):"
WEB_ON = (
    "You can search the web with web_search and read pages with web_fetch. Use them whenever you need current, "
    "specific or niche information you are not sure about, or when you cannot complete the task from what you know "
    "(for example an unfamiliar API, an error message, a package version or recent events). Search first, read the "
    "most relevant page, then answer and cite the sources you used as Markdown links."
)
WEB_OFF = (
    "Internet access is turned off, so you cannot look things up. If a good answer needs online information, say so "
    "briefly and tell the user they can turn on Web in the header."
)
DEFAULT_MAX_STEPS = 30
AUTO_MAX_STEPS = 150
# How often per turn the agent pushes on when the model stops with work left over.
CONTINUATIONS = {"auto": 12, "agent": 3}
_UNFINISHED = re.compile(
    r"\b(next steps?|the next step|next,? (?:i|we)(?:'ll| will)|i will now|i'll now|let'?s (?:now )?(?:proceed|continue|implement|create|add)|"
    r"still (?:need|needs|to do)|remaining (?:steps|tasks|work|features)|to be implemented|not yet implemented|todo:)\b",
    re.IGNORECASE,
)
_ASKS_USER_TO_CHOOSE = re.compile(
    r"(please (?:choose|select|pick|confirm|decide|let me know)|which (?:option|one|approach) (?:would|do) you|"
    r"would you like (?:me )?to|do you want (?:me )?to|should i (?:proceed|continue|go ahead)|choose an option)",
    re.IGNORECASE,
)
TOOL_RESULT_CHARS = 12_000
OLD_TOOL_RESULT_CHARS = 1_500
LIVE_EVENT_LIMIT = 4000
SLASH_COMMANDS = {
    "/help": "Show the available commands.",
    "/clear": "Start over with an empty context (the transcript is kept).",
    "/compact": "Summarise the conversation so far to free up context.",
    "/model": "/model <name> switches the model for this session; /model lists models.",
    "/mode": "/mode ask|plan|agent|auto switches the mode.",
    "/init": "Explore the project and write an AGENTS.md for future sessions.",
    "/undo": "Revert the file edits made in the last turn.",
}
_SHELL_META = re.compile(r"[;&|`]|\$\(|>|<")


class AgentError(Exception):
    """User-facing error (bad request, no model, …)."""


@dataclass
class _Pending:
    call_id: str
    event: threading.Event = field(default_factory=threading.Event)
    decision: Optional[str] = None  # approve | deny
    remember: bool = False


@dataclass
class _Run:
    session_id: str
    turn: int
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    finished: bool = False

    def active(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and not self.finished)
    pending: dict[str, _Pending] = field(default_factory=dict)


class EventHub:
    """Persists events and keeps a live in-memory tail for streaming clients."""

    def __init__(self, store: AgentStore) -> None:
        self.store = store
        self._lock = threading.Condition()
        self._live: dict[str, list[dict[str, Any]]] = {}
        self._seq: dict[str, int] = {}

    def _next_seq(self, session_id: str) -> int:
        if session_id not in self._seq:
            events = self.store.list_events(session_id)
            self._seq[session_id] = events[-1]["seq"] if events else 0
        self._seq[session_id] += 1
        return self._seq[session_id]

    def publish(self, session_id: str, turn: int, event_type: str, payload: dict[str, Any], persist: bool = True) -> int:
        with self._lock:
            seq = self._next_seq(session_id)
            if persist:
                self.store.db.execute(
                    "INSERT INTO agent_events (session_id, seq, turn, type, payload, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                    (session_id, seq, turn, event_type, json.dumps(payload)),
                )
            event = {"seq": seq, "turn": turn, "type": event_type, **payload}
            live = self._live.setdefault(session_id, [])
            live.append(event)
            if len(live) > LIVE_EVENT_LIMIT:
                del live[: len(live) - LIVE_EVENT_LIMIT]
            self._lock.notify_all()
            return seq

    def live_after(self, session_id: str, after: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._live.get(session_id, []) if e["seq"] > after]

    def wait(self, timeout: float) -> None:
        with self._lock:
            self._lock.wait(timeout)

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._live.pop(session_id, None)
            self._seq.pop(session_id, None)


def command_prefix(command: str) -> Optional[str]:
    """The allow-list key for "always allow" (``python -m pytest``, ``npm test``, ``pytest``)."""
    if _SHELL_META.search(command):
        return None
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return None
    if not tokens:
        return None
    if len(tokens) > 2 and tokens[1] == "-m":
        return " ".join(tokens[:3]).lower()
    if len(tokens) > 1 and not tokens[1].startswith("-"):
        return " ".join(tokens[:2]).lower()
    return tokens[0].lower()


def needs_approval(mode: str, tool: AgentTool, args: Any, allowed: list[str]) -> tuple[bool, str]:
    """(needs approval?, risk) — risk is ``none``, ``edit``, ``command`` or ``destructive``."""
    if tool.kind in ("read", "meta", "network"):
        return False, "none"
    if tool.kind == "edit":
        return mode != "auto", "edit"
    command = getattr(args, "command", None) or ("git commit" if tool.name == "git_commit" else "")
    if command and is_destructive(command):
        return True, "destructive"
    if mode == "auto":
        return False, "command"  # Auto: only destructive commands still ask (above)
    prefix = command_prefix(command) if command else None
    if prefix and prefix in allowed:
        return False, "command"
    return True, "command"


class AgentService:
    def __init__(
        self,
        store: AgentStore,
        registry: Any,
        settings_store: SettingsStore,
        audit: AuditLog,
        prompts: PromptLibrary,
        workspace_factory: Optional[Callable[[str, bool], Workspace]] = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.settings_store = settings_store
        self.audit = audit
        self.prompts = prompts
        self.hub = EventHub(store)
        self._runs: dict[str, _Run] = {}
        self._lock = threading.RLock()
        self._workspace_factory = workspace_factory or (lambda root, internet: Workspace(root, internet_enabled=internet))

    # ============================================================ sessions
    @staticmethod
    def resolve_folder(workspace: str) -> Path:
        """User-typed folder → absolute existing directory, or a clear AgentError.

        Accepts what Windows hands out: Explorer's "Copy as path" wraps the path in quotes,
        and pasted text often has stray spaces. Relative paths are refused (they would be
        resolved against the backend's own folder).
        """
        cleaned = (workspace or "").strip().strip('"').strip("'").strip()
        if not cleaned:
            raise AgentError("Enter a folder path, for example C:\\Users\\you\\projects\\my-app.")
        expanded = Path(cleaned).expanduser()
        if not expanded.is_absolute():
            raise AgentError(f"Use a full path such as C:\\Users\\you\\projects\\my-app (got '{cleaned}').")
        root = canonicalize(expanded)
        if not root.exists():
            raise AgentError(f"Folder not found: {root}")
        if not root.is_dir():
            raise AgentError(f"That is a file, not a folder: {root}")
        return root

    def create_session(self, workspace: Optional[str], model_id: Optional[str] = None, mode: str = "auto") -> dict[str, Any]:
        if mode not in MODES:
            raise AgentError(f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}.")
        if mode == "chat":
            # Plain chat: no workspace, no file or command tools.
            session = self.store.create_session("", model_id or self._default_model(), "chat", title="New chat")
            return self.describe(session["id"])
        root = self.resolve_folder(workspace or "")
        self._approve_workspace(str(root))
        session = self.store.create_session(str(root), model_id or self._default_model(), mode)
        return self.describe(session["id"])

    def _approve_workspace(self, root: str) -> None:
        # Opening a folder in the UI is the user's approval of that workspace.
        settings = self.settings_store.load()
        if root not in settings.approved_workspaces:
            self.settings_store.save(
                settings.model_copy(update={"approved_workspaces": [*settings.approved_workspaces, root]})
            )

    def _default_model(self) -> Optional[str]:
        settings = self.settings_store.load()
        loaded = self.registry.runtime_info().get("loaded_models") or []
        return loaded[0] if loaded else settings.default_model_id

    def describe(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        with self._lock:
            run = self._runs.get(session_id)
            session["running"] = bool(run and run.active())
            session["pending_approvals"] = list(run.pending.keys()) if run else []
        return session

    def list_sessions(self, workspace: Optional[str] = None, kind: Optional[str] = None) -> list[dict[str, Any]]:
        """``kind``: ``chat`` (plain chats) or ``code`` (workspace sessions); None = all."""
        if kind == "chat":
            return [s for s in self.store.list_sessions() if s["mode"] == "chat"]
        key = str(canonicalize(workspace)) if workspace else None
        sessions = self.store.list_sessions(key)
        return [s for s in sessions if s["mode"] != "chat"] if kind == "code" else sessions

    def update_session(self, session_id: str, **fields: Any) -> dict[str, Any]:
        if "mode" in fields and fields["mode"] not in MODES:
            raise AgentError(f"Unknown mode '{fields['mode']}'.")
        if "mode" in fields:
            current = self.store.get_session(session_id) or {}
            if (current.get("mode") == "chat") != (fields["mode"] == "chat"):
                raise AgentError("A chat can't become a coding session (or the reverse). Start a new one instead.")
        self.store.update_session(session_id, **fields)
        return self.describe(session_id)

    def delete_session(self, session_id: str) -> None:
        self.interrupt(session_id)
        self.store.delete_session(session_id)
        self.hub.forget(session_id)

    def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        return self.store.list_events(session_id, after)

    # ============================================================ messages
    def send(self, session_id: str, text: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        text = text.strip()
        if not text:
            raise AgentError("Message is empty.")
        if self.is_running(session_id):
            raise AgentError("The agent is still working on the previous message. Wait or press Stop.")
        if text.startswith("/"):
            return self._slash(session, text)
        return self._start_turn(session, text)

    def is_running(self, session_id: str) -> bool:
        with self._lock:
            run = self._runs.get(session_id)
            return bool(run and run.active())

    def _start_turn(self, session: dict[str, Any], text: str, display: Optional[str] = None) -> dict[str, Any]:
        model_id = session.get("model_id")
        if not model_id:
            raise AgentError("No model selected for this session. Load a model and pick it with /model or the model menu.")
        if not self.registry.adapter.is_loaded(model_id):
            status = self.registry.status(model_id)
            reason = status.get("error") if status.get("state") == "crashed" else None
            raise AgentError(reason or "The selected model is not loaded. Load it in the Models panel first.")
        turn = self.store.last_turn(session["id"]) + 1
        run = _Run(session["id"], turn)
        with self._lock:
            self._runs[session["id"]] = run
        self.hub.publish(session["id"], turn, "turn_start", {"mode": session["mode"], "model_id": model_id})
        self.hub.publish(session["id"], turn, "user", {"text": display or text})
        self.store.add_message(session["id"], turn, {"role": "user", "content": text})
        if session["title"] in ("New session", "New chat"):
            title = (display or text).splitlines()[0][:60]
            self.store.update_session(session["id"], title=title)
        self.store.update_session(session["id"], state="running")
        run.thread = threading.Thread(target=self._run_turn, args=(run,), name=f"agent-{session['id'][:8]}", daemon=True)
        run.thread.start()
        return {"turn": turn, "started": True}

    # ============================================================ control
    def interrupt(self, session_id: str) -> bool:
        with self._lock:
            run = self._runs.get(session_id)
        if run is None or not run.active():
            return False
        run.cancel.set()
        for pending in list(run.pending.values()):
            pending.decision = "deny"
            pending.event.set()
        return True

    def decide(self, session_id: str, call_id: str, approve: bool, remember: bool = False) -> bool:
        """The ONLY way a pending action gets approved: an explicit user request."""
        with self._lock:
            run = self._runs.get(session_id)
        if run is None or call_id not in run.pending:
            return False
        pending = run.pending[call_id]
        pending.decision = "approve" if approve else "deny"
        pending.remember = bool(remember)
        pending.event.set()
        return True

    def wait_idle(self, session_id: str, timeout: float = 30.0) -> bool:
        with self._lock:
            run = self._runs.get(session_id)
        if run is None or run.thread is None:
            return True
        run.thread.join(timeout)
        return not run.thread.is_alive()

    # ============================================================ the loop
    def _variant(self, model_id: str) -> dict[str, Any]:
        record = self.registry.get_model(model_id) or {}
        if record.get("runtime") == "cloud":
            # API models (OpenAI, Claude, DeepSeek) all do native tool calls well.
            return {"tool_format": "native", "temperature": 0.3}
        return self.prompts.variant_for(self._model_name(model_id))

    def _model_name(self, model_id: str) -> str:
        record = self.registry.get_model(model_id) or {}
        return Path(record.get("path") or "").name or record.get("name") or model_id

    def _run_turn(self, run: _Run) -> None:
        session_id, turn = run.session_id, run.turn
        reason = "completed"
        try:
            reason = self._loop(run)
        except Exception as exc:  # noqa: BLE001 - always end the turn cleanly
            log.exception("Agent turn failed")
            self.hub.publish(session_id, turn, "error", {"message": _friendly(exc)})
            reason = "error"
        finally:
            if run.cancel.is_set() and reason == "completed":
                reason = "interrupted"
            self.store.update_session(session_id, state="idle")
            with self._lock:
                run.pending.clear()
                run.finished = True
            self.hub.publish(session_id, turn, "turn_end", {"reason": reason})

    def _loop(self, run: _Run) -> str:
        session = self.store.get_session(run.session_id)
        assert session is not None
        model_id = session["model_id"]
        mode = session["mode"]
        settings = self.settings_store.load()
        variant = self._variant(model_id)
        native = self._use_native(model_id, variant)
        tools = tools_for_mode(mode, settings.internet_enabled)
        ws = self._workspace_factory(session["workspace"] or str(Path.home()), settings.internet_enabled)
        ws.todos = list(session.get("todos") or [])
        ws.cancel = run.cancel
        options = dict(session.get("options") or {})
        max_steps = int(options.get("max_steps") or (AUTO_MAX_STEPS if mode == "auto" else DEFAULT_MAX_STEPS))
        n_ctx = self._context_size(model_id)
        max_gen = max(512, min(int(settings.inference.max_tokens or 4096), n_ctx // 3, 8192))
        temperature = float(options.get("temperature", variant.get("temperature", 0.3)))
        repeats: dict[str, int] = {}
        stuck = 0  # consecutive steps whose every call was a refused repeat
        empty_steps = 0  # consecutive steps where the model produced reasoning but no action/answer
        used_tools = False
        # Constrained mode: every step is a JSON-schema-constrained tool call (or "respond").
        constrained = variant.get("tool_format") == "json" and mode != "chat"
        escalated = False
        continuations = 0
        last_final = ""
        # After the agent pushes the model back to work, it must act (edit / run) before it may
        # answer again; in constrained mode "respond" is removed from the grammar meanwhile.
        force_action = False
        last_push_id: Optional[int] = None
        # Chat with web access on a model that won't emit real tool calls: each step first makes a
        # grammar-forced choice (web_search / web_fetch / answer); the answer itself then streams.
        chat_lookup = mode == "chat" and variant.get("tool_format") == "json" and any(t.kind == "network" for t in tools)
        lookups = 0
        answering = False
        last_command: Optional[dict[str, Any]] = None  # {"command", "exit_code"} of the latest run_command
        if not native and variant.get("tool_format") not in ("text", "json") and run.turn == 1:
            self.hub.publish(run.session_id, run.turn, "notice", {"text": "This model's chat template has no native tool calling; using the text tool protocol."})

        for step in range(max_steps):
            if run.cancel.is_set():
                return "interrupted"
            fmt = "json" if constrained else ("native" if native else "text")
            system = self._system_prompt(session["workspace"], mode, tools, fmt)
            history = self._history(run.session_id, native and not constrained)
            history = self._fit_context(run, model_id, system, history, n_ctx, max_gen, native and not constrained)
            messages = [{"role": "system", "content": system}] + history
            # Raise the temperature while the model is looping on the same call.
            step_temp = min(1.0, temperature + 0.3 * stuck) if stuck else temperature
            call_options = {"temperature": step_temp, "max_tokens": max_gen, "top_p": 0.95, "repeat_penalty": 1.05, "effort": "medium" if mode == "chat" else "high"}
            if constrained:
                result = self._call_constrained(run, model_id, messages, call_options, tools, allow_respond=not force_action)
            elif chat_lookup and not answering:
                result = self._call_constrained(
                    run, model_id, messages, {**call_options, "max_tokens": 512}, tools, allow_respond=False, answer_option=True
                )
                if result is not None and (not result[2] or result[2][0].get("name") == "answer" or lookups >= 4):
                    answering = True  # enough looked up: write the answer (streamed) next
                    continue
                lookups += 1
            else:
                if native and tools and not chat_lookup:
                    call_options["tools"] = [t.openai() for t in tools]
                if answering:
                    # Not stored: steer the final, streamed answer onto the lookup results.
                    messages = messages + [{"role": "user", "content": (
                        "Now answer my original question directly, using the lookup results above where they help. "
                        "Cite the pages you used as Markdown links. Do not call tools and do not ask me questions."
                    )}]
                result = self._call_model(run, model_id, messages, call_options, native, {t.name for t in tools})
            if result is None:
                return "interrupted"
            content, reasoning, calls, usage = result
            self._publish_usage(run, model_id, usage, n_ctx)
            if content.strip() or calls:
                empty_steps = 0

            if not calls and not constrained and not escalated and self._should_nudge(mode, content, used_tools):  # noqa: E501
                # The model described/pasted the work instead of doing it. Show what it said,
                # keep it out of the context, and redo the step with constrained tool calls.
                escalated = constrained = True
                if content.strip() or reasoning.strip():
                    self.hub.publish(run.session_id, run.turn, "assistant", {"content": content, "reasoning": reasoning})
                self.hub.publish(
                    run.session_id, run.turn, "notice",
                    {"text": "The model described the work instead of doing it; switching to constrained tool calls for this turn."},
                )
                continue

            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            if native and calls:
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in calls
                ]
            elif calls:
                # Text protocol: keep the call in the content so the model sees what it did.
                assistant["content"] = (content + "\n" if content else "") + "\n".join(
                    "<tool_call>\n" + json.dumps({"name": c["name"], "arguments": c["arguments"]}) + "\n</tool_call>" for c in calls
                )
            assistant_id = self.store.add_message(run.session_id, run.turn, assistant)
            if content.strip() or reasoning.strip():
                self.hub.publish(run.session_id, run.turn, "assistant", {"content": content, "reasoning": reasoning})

            if not calls:
                if not content.strip():
                    # The model emitted reasoning (or nothing) without acting or answering.
                    # Auto-continue instead of ending the turn and making the user say "continue".
                    empty_steps += 1
                    if empty_steps >= 3:
                        self.hub.publish(
                            run.session_id, run.turn, "notice",
                            {"text": "Stopped: the model kept producing reasoning without an action or answer. Try rephrasing, a different model, or /compact."},
                        )
                        return "stuck"
                    self.store.add_message(
                        run.session_id, run.turn,
                        {"role": "user", "content": "Continue: state your next tool call, or give your final answer if you are done."},
                    )
                    self.hub.publish(
                        run.session_id, run.turn, "notice",
                        {"text": "The model paused after reasoning; continuing automatically."},
                    )
                    continue
                problem = self._completion_problem(mode, used_tools, last_command, ws.todos, content, last_final)
                last_final = content.strip()
                if problem and continuations < CONTINUATIONS.get(mode, 0):
                    # The model stopped with work left over: keep going instead of making the
                    # user type "continue" (bounded per turn; Ask/Plan answers are never pushed).
                    continuations += 1
                    # Keep the status reply and older pushes out of the model's context (the user
                    # still sees them): small models otherwise copy their own last answer.
                    self.store.deactivate_message(assistant_id)
                    if last_push_id is not None:
                        self.store.deactivate_message(last_push_id)
                    if mode == "auto":
                        force_action = True
                        if not constrained and not native:
                            constrained = True  # text-protocol models: enforce the next action by grammar
                        problem += (
                            " First make sure your todo list covers the user's whole request (add the missing items),"
                            " then do the next item."
                        )
                    last_push_id = self.store.add_message(run.session_id, run.turn, {"role": "user", "content": problem})
                    self.hub.publish(
                        run.session_id, run.turn, "notice",
                        {"text": f"The model stopped with work left; continuing automatically ({continuations}/{CONTINUATIONS.get(mode, 0)})."},
                    )
                    continue
                return "completed"
            used_tools = True
            if any(TOOLS_BY_NAME.get(c.get("name") or "") is not None and TOOLS_BY_NAME[c["name"]].kind in ("edit", "command") for c in calls):
                force_action = False  # it acted; it may answer again later
            refused = 0
            for call in calls:
                if run.cancel.is_set():
                    return "interrupted"
                signature = call["name"] + json.dumps(call["arguments"], sort_keys=True)
                repeats[signature] = repeats.get(signature, 0) + 1
                limit = 1 if call["name"] in ("edit_file", "write_file", "web_search", "web_fetch") else 2
                if repeats[signature] > limit:
                    refused += 1
                    output = (
                        "Refused: you already made this exact call in this turn"
                        + (" and it was applied — the file has already changed. Read the file again before editing."
                           if call["name"] in ("edit_file", "write_file") else ".")
                        + " Do something different, or reply to the user if you are stuck."
                    )
                    self._tool_message(run, native, call, output)
                    continue
                output = self._execute(run, session, mode, ws, call, tools)
                self._tool_message(run, native, call, output)
                match = re.match(r"Exit code (-?\d+)", output) if call["name"] == "run_command" else None
                if match:  # only commands that actually ran (not denied/invalid ones)
                    last_command = {
                        "command": str(call["arguments"].get("command", "")),
                        "exit_code": int(match.group(1)),
                        "status": "ok" if match.group(1) == "0" else "failed",
                    }
            stuck = stuck + 1 if refused == len(calls) else 0
            if stuck >= 3 and mode == "auto" and continuations < CONTINUATIONS["auto"]:
                # Unattended: redirect once more (counts as a continuation) before giving up.
                continuations += 1
                stuck = 0
                self.store.add_message(run.session_id, run.turn, {"role": "user", "content": (
                    "You are stuck repeating the same action. Stop and re-plan: read the current state of the files "
                    "you are working on (read_file) and the last error, then take a different next step."
                )})
                self.hub.publish(
                    run.session_id, run.turn, "notice",
                    {"text": f"The model was repeating itself; redirected it ({continuations}/{CONTINUATIONS['auto']})."},
                )
            elif stuck >= 3:
                self.hub.publish(
                    run.session_id, run.turn, "notice",
                    {"text": "Stopped: the model kept repeating the same action. Try rephrasing the task, a larger model, or /compact."},
                )
                return "stuck"
            if ws.todos != (session.get("todos") or []):
                self.store.update_session(run.session_id, todos=ws.todos)
                session["todos"] = ws.todos
                self.hub.publish(run.session_id, run.turn, "todos", {"todos": ws.todos})
        self.hub.publish(
            run.session_id, run.turn, "notice",
            {"text": f"Stopped after {max_steps} steps (step budget). Send a message to continue."},
        )
        return "step_budget"

    @staticmethod
    def _completion_problem(
        mode: str,
        used_tools: bool,
        last_command: Optional[dict[str, Any]],
        todos: list[dict[str, Any]],
        content: str = "",
        previous_final: str = "",
    ) -> Optional[str]:
        """What is visibly unfinished when the model tries to end an Agent/Auto turn.

        Returns the message that sends it back to work, or None when it may stop.
        """
        if mode not in ("agent", "auto"):
            return None
        text = content.strip()
        if text and previous_final and text == previous_final:
            return (
                "You repeated your previous message word for word. Do not report status again — "
                "call a tool now to make the next concrete change."
            )
        if mode == "auto" and _ASKS_USER_TO_CHOOSE.search(text):
            return (
                "You are running autonomously: do not ask the user to choose. Pick the most sensible option "
                "yourself, say in one line which one and why, and carry on with the tools."
            )
        if not used_tools:
            return None
        problems = []
        if _UNFINISHED.search(text):
            problems.append("Your reply says there is still work to do.")
        if last_command and last_command["status"] == "failed":
            problems.append(
                f"The last command you ran (`{last_command['command']}`) failed with exit code {last_command['exit_code']}."
            )
        open_items = [t["content"] for t in todos if t.get("status") != "completed"]
        if open_items:
            problems.append("Your todo list still has open items: " + "; ".join(open_items[:6]) + ".")
        if not problems:
            return None
        return (
            " ".join(problems)
            + " Do not stop to report progress: continue now with the next step using the tools, verify it"
            " (re-run the command), and keep the todo list up to date. Reply in text only when everything is"
            " done, or if you are truly blocked — then say exactly what still fails and why."
        )

    @staticmethod
    def _should_nudge(mode: str, content: str, used_tools: bool) -> bool:
        """Small models often answer an agent task with a plan or pasted code and stop.

        Escalate once per turn (constrained mode then stays on), only in Agent/Auto mode, and
        only when the reply looks like undone work (a code block or a numbered plan).
        """
        if mode not in ("agent", "auto"):
            return False
        looks_like_work = "```" in content or bool(re.search(r"(?m)^\s*(?:#+\s*)?(?:\d+\.|step \d)", content, re.I))
        return looks_like_work

    # ------------------------------------------------------------ prompts
    def _use_native(self, model_id: str, variant: dict[str, Any]) -> bool:
        fmt = variant.get("tool_format", "auto")
        if fmt == "text":
            return False
        supported = self.registry.adapter.supports_native_tools(model_id)
        return supported if fmt == "auto" else (fmt == "native" and supported)

    def _system_prompt(self, workspace: str, mode: str, tools: list[AgentTool], fmt: str) -> str:
        """``fmt``: ``native`` (OpenAI tools), ``text`` (<tool_call> blocks) or ``json`` (constrained)."""
        web = any(t.kind == "network" for t in tools)
        if mode == "chat":
            from datetime import date

            from .environment import os_name

            values: dict[str, Any] = {"date": date.today().isoformat(), "os": os_name(), "web_instructions": WEB_ON if web else WEB_OFF}
            values["tool_instructions"] = self._tool_instructions(tools, fmt) if tools else ""
            return self.prompts.render(self.prompts.get("chat.md"), values)
        values = environment_values(workspace)
        values["mode_instructions"] = self.prompts.mode_instructions(mode)
        values["web_hint"] = ("When you are missing information: " + WEB_ON) if web else WEB_OFF
        values["tool_instructions"] = self._tool_instructions(tools, fmt)
        return self.prompts.render(self.prompts.get("system.md"), values)

    def _tool_instructions(self, tools: list[AgentTool], fmt: str) -> str:
        values: dict[str, Any] = {}
        if fmt == "json":
            catalog = "\n".join(
                f"- {t.name}: {t.description}\n  arguments: {json.dumps(t.schema().get('properties', {}))}" for t in tools
            )
            values["tool_instructions"] = self.prompts.render(self.prompts.get("tools_json.md"), {"tool_catalog": catalog})
        elif fmt == "native":
            summary = "\n".join(f"- {t.name}: {t.description}" for t in tools)
            values["tool_instructions"] = self.prompts.render(self.prompts.get("tools_native.md"), {"tool_summary": summary})
        else:
            catalog = "\n".join(
                f"- {t.name}: {t.description}\n  arguments: {json.dumps(t.schema().get('properties', {}))}" for t in tools
            )
            values["tool_instructions"] = self.prompts.render(self.prompts.get("tools_text.md"), {"tool_catalog": catalog})
        return values["tool_instructions"]

    def preview_system_prompt(self, session_id: str) -> str:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        settings = self.settings_store.load()
        variant = self._variant(session["model_id"]) if session.get("model_id") else {}
        if variant.get("tool_format") == "json" and session["mode"] != "chat":
            fmt = "json"
        elif session.get("model_id") and self._use_native(session["model_id"], variant):
            fmt = "native"
        else:
            fmt = "text"
        return self._system_prompt(session["workspace"], session["mode"], tools_for_mode(session["mode"], settings.internet_enabled), fmt)

    def _history(self, session_id: str, native: bool) -> list[dict[str, Any]]:
        stored = self.store.messages(session_id)
        # Drop older copies of identical status replies (assistant text without tool calls):
        # small models imitate their own repeated answers. The newest copy is kept.
        seen: set[str] = set()
        duplicates: set[int] = set()
        for message in reversed(stored):
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                key = str(message.get("content") or "").strip()
                if key and key in seen:
                    duplicates.add(message["_id"])
                seen.add(key)
        out = []
        for message in stored:
            if message["_id"] in duplicates:
                continue
            msg = {k: v for k, v in message.items() if not k.startswith("_")}
            if not native:
                if msg.get("role") == "tool":
                    content = str(msg.get("content", ""))
                    msg = {"role": "user", "content": content if content.startswith("Tool result") else f"Tool result:\n{content}"}
                calls = msg.pop("tool_calls", None)
                if calls:
                    # Keep what was called visible to models that do not use native tool messages.
                    lines = [json.dumps({"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}) for c in calls]
                    msg["content"] = ((msg.get("content") or "") + "\n" + "\n".join(lines)).strip()
            out.append(msg)
        return out

    def _tool_message(self, run: _Run, native: bool, call: dict[str, Any], output: str) -> None:
        text = output if len(output) <= TOOL_RESULT_CHARS else output[:TOOL_RESULT_CHARS] + "\n… [output truncated]"
        if native:
            message = {"role": "tool", "tool_call_id": call["id"], "content": text}
        else:
            message = {"role": "tool", "content": f"Tool result for {call['name']}:\n{text}"}
        self.store.add_message(run.session_id, run.turn, message)

    # ------------------------------------------------------------ context
    def _context_size(self, model_id: str) -> int:
        status = self.registry.status(model_id) or {}
        value = status.get("n_ctx_effective") or status.get("n_ctx") or self.settings_store.load().inference.n_ctx or 8192
        return int(value)

    @staticmethod
    def _estimate(messages: list[dict[str, Any]], chars_per_token: float = 3.5) -> int:
        chars = 0
        for m in messages:
            chars += len(str(m.get("content") or "")) + 16
            for call in m.get("tool_calls") or []:
                chars += len(call["function"]["arguments"]) + 32
        return int(chars / chars_per_token)

    def _fit_context(
        self,
        run: _Run,
        model_id: str,
        system: str,
        history: list[dict[str, Any]],
        n_ctx: int,
        max_gen: int,
        native_history: bool = True,
    ) -> list[dict[str, Any]]:
        budget = n_ctx - max_gen - 256
        system_cost = self._estimate([{"content": system}])
        if system_cost + self._estimate(history) <= budget * 0.85:
            return history
        # 1) Shrink old tool results (only the newest two messages stay complete).
        self._shrink_old_outputs(history)
        if system_cost + self._estimate(history) <= budget * 0.85:
            return history
        # 2) Summarise the middle of the conversation with the model (compaction). Works inside
        #    one long turn too: the task message and the newest messages (up to about half the
        #    budget) stay verbatim, so the next compaction is many steps away.
        keep, used = 0, 0
        for message in reversed(history):
            used += self._estimate([message])
            if keep >= 2 and used > budget * 0.5:
                break
            keep += 1
        summary = self.compact(run.session_id, model_id=model_id, turn=run.turn, keep_recent=keep)
        if summary:
            history = self._history(run.session_id, native_history)
            self._shrink_old_outputs(history)
        # 3) Last resort: drop the oldest messages, but never the summary or the user's task.
        protected = 0
        if history and str(history[0].get("content") or "").startswith(SUMMARY_PREFIX):
            protected = 2  # the summary and the model's acknowledgement
        if protected < len(history) and history[protected].get("role") == "user":
            protected += 1  # the task, kept verbatim
        while len(history) > protected + 2 and system_cost + self._estimate(history) > budget:
            history.pop(protected)
        return history

    @staticmethod
    def _shrink_old_outputs(history: list[dict[str, Any]]) -> None:
        for message in history[:-2]:
            content = str(message.get("content") or "")
            if len(content) > OLD_TOOL_RESULT_CHARS and (message.get("role") == "tool" or content.startswith("Tool result")):
                message["content"] = content[:OLD_TOOL_RESULT_CHARS] + "\n… [older output shortened]"

    def compact(
        self, session_id: str, model_id: Optional[str] = None, turn: int = 0, keep_recent: Optional[int] = None
    ) -> str:
        """Summarise the conversation so the model keeps working with a short context.

        ``keep_recent=None`` (/compact): everything is summarised. Otherwise the current
        turn's task message and the last ``keep_recent`` messages stay verbatim and only the
        part in between is summarised — so this also works inside one long autonomous turn.
        Returns the summary, or "" when there was nothing to summarise.
        """
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        model_id = model_id or session.get("model_id")
        rows = self.store.messages(session_id)
        task: Optional[dict[str, Any]] = None
        tail: list[dict[str, Any]] = []
        if keep_recent is not None and rows:
            keep = max(2, min(keep_recent, len(rows) - 1))
            start = len(rows) - keep
            # Never start the kept tail with a tool result whose call would be summarised away.
            while start > 0 and rows[start].get("role") == "tool":
                start -= 1
            tail = rows[start:]
            current = rows[-1]["_turn"]
            task = next(
                (
                    r for r in rows[:start]
                    if r["_turn"] == current and r.get("role") == "user" and not str(r.get("content") or "").startswith(SUMMARY_PREFIX)
                ),
                None,
            )
            older = [r for r in rows[:start] if r is not task]
        else:
            older = rows
        if not older:
            return ""
        transcript = []
        for r in older:
            role = r.get("role")
            content = str(r.get("content") or "")
            if r.get("tool_calls"):
                content += " " + " ".join(f"[called {c['function']['name']}({c['function']['arguments'][:300]})]" for c in r["tool_calls"])
            transcript.append(f"{role.upper()}: {content[:4000]}")
        prompt = self.prompts.render(self.prompts.get("compact.md"), {})
        messages = [
            {"role": "system", "content": "You summarise coding-agent sessions accurately and concisely."},
            {"role": "user", "content": prompt + "\n\n--- Conversation ---\n" + "\n\n".join(transcript)[-60_000:]},
        ]
        summary = strip_think(self.registry.adapter.chat(model_id, messages, {"temperature": 0.2, "max_tokens": 1024})["content"])
        # Summary first, then the task (if kept) and the recent messages, verbatim.
        self.store.deactivate_messages(session_id)
        self.store.add_message(session_id, turn, {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"})
        self.store.add_message(session_id, turn, {"role": "assistant", "content": "Understood. I will continue from this summary."})
        for r in ([task] if task else []) + tail:
            self.store.add_message(session_id, r["_turn"], {k: v for k, v in r.items() if not k.startswith("_")})
        self.hub.publish(
            session_id, turn, "notice",
            {"text": f"Context was getting full; summarised {len(older)} older messages." if keep_recent is not None else "Conversation compacted.", "summary": summary},
        )
        return summary

    def _publish_usage(self, run: _Run, model_id: str, usage: dict[str, Any], n_ctx: int) -> None:
        timings = (self.registry.status(model_id) or {}).get("last_timings") or {}
        prompt = usage.get("prompt_tokens") or timings.get("prompt_n")
        completion = usage.get("completion_tokens") or timings.get("predicted_n")
        used = (prompt or 0) + (completion or 0)
        self.hub.publish(
            run.session_id, run.turn, "usage",
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "context_used": used or None,
                "context_max": n_ctx,
                "tokens_per_second": timings.get("predicted_per_second"),
                "prompt_tokens_per_second": timings.get("prompt_per_second"),
            },
        )

    # ------------------------------------------------------------ model call
    def _call_model(
        self, run: _Run, model_id: str, messages: list[dict[str, Any]], options: dict[str, Any], native: bool, known: set[str]
    ) -> Optional[tuple[str, str, list[dict[str, Any]], dict[str, Any]]]:
        splitter = StreamSplitter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        native_calls: dict[int, dict[str, Any]] = {}
        text_blocks: list[str] = []
        usage: dict[str, Any] = {}
        stream = self.registry.adapter.stream_events(model_id, messages, options)
        try:
            for event in stream:
                if run.cancel.is_set():
                    return None
                kind = event["type"]
                if kind == "reasoning":
                    reasoning_parts.append(event["text"])
                    self.hub.publish(run.session_id, run.turn, "delta", {"channel": "reasoning", "text": event["text"]}, persist=False)
                elif kind == "content":
                    for part, text in splitter.feed(event["text"]):
                        self._route_part(run, part, text, content_parts, reasoning_parts, text_blocks)
                elif kind == "tool_call":
                    slot = native_calls.setdefault(event.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if event.get("id"):
                        slot["id"] = event["id"]
                    if event.get("name"):
                        slot["name"] = event["name"]
                    slot["arguments"] += event.get("arguments") or ""
                    self.hub.publish(
                        run.session_id, run.turn, "delta",
                        {"channel": "tool", "name": slot["name"], "chars": len(slot["arguments"])}, persist=False,
                    )
                elif kind == "done":
                    usage = event.get("usage") or {}
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        for part, text in splitter.close():
            self._route_part(run, part, text, content_parts, reasoning_parts, text_blocks)

        calls: list[dict[str, Any]] = []
        for index in sorted(native_calls):
            slot = native_calls[index]
            try:
                args = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                args = parse_tool_block('{"name": "x", "arguments": ' + (slot["arguments"] or "{}") + "}").get("arguments", {})
            calls.append({"id": slot["id"] or f"call_{uuid.uuid4().hex[:8]}", "name": slot["name"], "arguments": args})
        content = "".join(content_parts).strip()
        if not calls:
            for block in text_blocks:
                parsed = parse_tool_block(block)
                if "error" in parsed:
                    calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": "_invalid", "arguments": {}, "error": parsed["error"]})
                else:
                    calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", **parsed})
        if not calls:
            rescued, rest = rescue_tool_calls(content, known)
            if rescued:
                content = rest
                for parsed in rescued:
                    calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", **parsed})
        if not calls and not native:
            for parsed in find_fenced_tool_calls(content, known):
                calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", **parsed})
        return content, "".join(reasoning_parts).strip(), calls, usage

    @staticmethod
    def constrained_schema(tools: list[AgentTool], allow_respond: bool = True, answer_option: bool = False) -> dict[str, Any]:
        """JSON schema for one step: exactly one tool call, or ``respond`` to finish.

        ``allow_respond=False`` (after the agent pushed the model back to work) leaves only
        the tools, so the model cannot answer with another status message.

        llama.cpp turns this into a grammar, so the output is always a valid call.
        """
        defs: dict[str, Any] = {}
        options = []
        # A short scratchpad before acting (ReAct style) noticeably helps small models.
        thought = {"type": "string", "description": "Brief reasoning: what you learned and why this is the next step."}
        for tool in tools:
            schema = tool.schema()
            defs.update(schema.pop("$defs", {}))
            options.append({
                "type": "object",
                "properties": {"thought": thought, "name": {"const": tool.name}, "arguments": schema},
                "required": ["thought", "name", "arguments"],
            })
        if allow_respond:
            options.append({
                "type": "object",
                "properties": {
                    "thought": thought,
                    "name": {"const": "respond"},
                    "arguments": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                },
                "required": ["thought", "name", "arguments"],
            })
        if answer_option:
            # Chat: "answer" ends the lookups; the answer is then generated (and streamed) normally.
            options.append({
                "type": "object",
                "properties": {"thought": thought, "name": {"const": "answer"}, "arguments": {"type": "object", "properties": {}}},
                "required": ["thought", "name", "arguments"],
            })
        schema: dict[str, Any] = {"oneOf": options}
        if defs:
            schema["$defs"] = defs
        return schema

    def _call_constrained(
        self,
        run: _Run,
        model_id: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[AgentTool],
        allow_respond: bool = True,
        answer_option: bool = False,
    ) -> Optional[tuple[str, str, list[dict[str, Any]], dict[str, Any]]]:
        schema = self.constrained_schema(tools, allow_respond=allow_respond, answer_option=answer_option)
        options = {**options, "response_format": {"type": "json_schema", "json_schema": {"name": "agent_step", "schema": schema}}}
        parts: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, Any] = {}
        stream = self.registry.adapter.stream_events(model_id, messages, options)
        try:
            for event in stream:
                if run.cancel.is_set():
                    return None
                if event["type"] == "content":
                    parts.append(event["text"])
                    self.hub.publish(
                        run.session_id, run.turn, "delta",
                        {"channel": "tool", "name": "", "chars": sum(len(p) for p in parts)}, persist=False,
                    )
                elif event["type"] == "reasoning":
                    reasoning.append(event["text"])
                    self.hub.publish(run.session_id, run.turn, "delta", {"channel": "reasoning", "text": event["text"]}, persist=False)
                elif event["type"] == "done":
                    usage = event.get("usage") or {}
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        raw = strip_think("".join(parts))
        parsed = parse_tool_block(raw)
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        try:
            thought_text = str(json.loads(raw).get("thought") or "")
        except (json.JSONDecodeError, AttributeError):
            thought_text = ""
        thinking = "\n\n".join(x for x in ("".join(reasoning).strip(), thought_text.strip()) if x)
        if "error" in parsed:
            return "", thinking, [{"id": call_id, "name": "_invalid", "arguments": {}, "error": parsed["error"]}], usage
        if parsed["name"] == "respond":
            return str(parsed["arguments"].get("message") or ""), thinking, [], usage
        return "", thinking, [{"id": call_id, **parsed}], usage

    def _route_part(self, run, part, text, content_parts, reasoning_parts, text_blocks) -> None:
        if part == "tool_block":
            text_blocks.append(text)
            return
        target = content_parts if part == "content" else reasoning_parts
        target.append(text)
        self.hub.publish(run.session_id, run.turn, "delta", {"channel": part, "text": text}, persist=False)

    # ------------------------------------------------------------ tools
    def _execute(
        self, run: _Run, session: dict[str, Any], mode: str, ws: Workspace, call: dict[str, Any], tools: list[AgentTool]
    ) -> str:
        call_id = call["id"]
        name = call.get("name") or ""
        base = {"call_id": call_id, "name": name, "args": call.get("arguments") or {}}
        if call.get("error"):
            self.hub.publish(run.session_id, run.turn, "tool", {**base, "kind": "invalid", "status": "error"})
            self.hub.publish(run.session_id, run.turn, "tool_update", {"call_id": call_id, "status": "error", "output": call["error"]})
            return f"Error: {call['error']}"
        tool = TOOLS_BY_NAME.get(name)
        if tool is None or tool not in tools:
            reason = (
                f"Tool '{name}' is not available in {mode} mode." if tool is not None else f"Unknown tool '{name}'."
            ) + f" Available tools: {', '.join(t.name for t in tools)}."
            self.hub.publish(run.session_id, run.turn, "tool", {**base, "kind": "invalid", "status": "error"})
            self.hub.publish(run.session_id, run.turn, "tool_update", {"call_id": call_id, "status": "error", "output": reason})
            return f"Error: {reason}"

        plan: Optional[EditPlan] = None
        try:
            args = tool.validate(base["args"])
            if tool.kind == "edit":
                plan = tool.prepare(ws, args)  # type: ignore[misc]
        except (ToolError, WorkspaceViolation, OSError) as exc:
            self.hub.publish(run.session_id, run.turn, "tool", {**base, "kind": tool.kind, "status": "error"})
            self.hub.publish(run.session_id, run.turn, "tool_update", {"call_id": call_id, "status": "error", "output": str(exc)})
            self._audit(session, tool, "error", base["args"], str(exc))
            return f"Error: {exc}"

        allowed = list((self.store.get_session(run.session_id) or {}).get("options", {}).get("allowed_commands", []))
        approval, risk = needs_approval(mode, tool, args, allowed)
        card = {**base, "kind": tool.kind, "risk": risk, "status": "awaiting_approval" if approval else "running"}
        if plan is not None:
            card.update({"diff": plan.diff, "path": plan.rel, "added": plan.added, "removed": plan.removed, "created": plan.before is None})
        if tool.kind == "command":
            card["command"] = getattr(args, "command", None) or f"git commit -m {getattr(args, 'message', '')!r}"
            card["prefix"] = command_prefix(card["command"]) if risk != "destructive" else None
        self.hub.publish(run.session_id, run.turn, "tool", card)

        if approval:
            pending = _Pending(call_id)
            with self._lock:
                run.pending[call_id] = pending
            self.store.update_session(run.session_id, state="awaiting_approval")
            while not pending.event.wait(0.25):
                if run.cancel.is_set():
                    pending.decision = "deny"
                    break
            with self._lock:
                run.pending.pop(call_id, None)
            self.store.update_session(run.session_id, state="running")
            if pending.decision != "approve":
                why = "interrupted" if run.cancel.is_set() else "denied by the user"
                self.hub.publish(run.session_id, run.turn, "tool_update", {"call_id": call_id, "status": "denied", "output": why})
                self._audit(session, tool, "denied", base["args"], why)
                return f"The user {'stopped the run' if run.cancel.is_set() else 'denied this action'}. Do not retry it; ask how to proceed."
            if pending.remember and card.get("prefix"):
                options = dict((self.store.get_session(run.session_id) or {}).get("options") or {})
                options["allowed_commands"] = sorted(set(options.get("allowed_commands", [])) | {card["prefix"]})
                self.store.update_session(run.session_id, options=options)
                self.hub.publish(run.session_id, run.turn, "notice", {"text": f"Commands starting with `{card['prefix']}` will run without asking in this session."})
            self.hub.publish(run.session_id, run.turn, "tool_update", {"call_id": call_id, "status": "running"})

        def on_output(text: str) -> None:
            self.hub.publish(run.session_id, run.turn, "tool_output", {"call_id": call_id, "text": text}, persist=False)

        ws.on_output = on_output
        try:
            if plan is not None:
                outcome = apply_plan(plan)
                self.store.add_checkpoint(run.session_id, run.turn, call_id, str(plan.path), plan.before, plan.after)
            else:
                outcome = tool.run(ws, args)  # type: ignore[misc]
        except (ToolError, WorkspaceViolation) as exc:
            outcome = ToolOutcome("error", f"Error: {exc}")
        except Exception as exc:  # noqa: BLE001 - tool crashes are reported to the model
            log.exception("Tool %s crashed", name)
            outcome = ToolOutcome("error", f"Error: {type(exc).__name__}: {exc}")
        finally:
            ws.on_output = None
        self.hub.publish(
            run.session_id, run.turn, "tool_update",
            {"call_id": call_id, "status": outcome.status, "output": outcome.output[:TOOL_RESULT_CHARS], "data": outcome.data},
        )
        self._audit(session, tool, outcome.status, base["args"], outcome.output)
        return outcome.output

    def _audit(self, session: dict[str, Any], tool: AgentTool, status: str, args: dict[str, Any], summary: str) -> None:
        level = {"read": "read_only", "meta": "read_only", "network": "read_only", "edit": "workspace_write"}.get(tool.kind, "system_action")
        target = args.get("path") or args.get("command") or args.get("url") or session["workspace"]
        try:
            self.audit.record_execution(
                tool_name=f"agent.{tool.name}", status=status, permission_level=level,
                target=str(target)[:500], result_summary=(summary or "")[:500], task_id=session["id"],
            )
        except Exception:  # noqa: BLE001 - auditing must not break the run
            log.exception("Audit write failed")

    # ============================================================ checkpoints
    def changes(self, session_id: str) -> list[dict[str, Any]]:
        """Changed files per turn with a combined diff (first before → last after)."""
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for cp in self.store.checkpoints(session_id):
            grouped.setdefault((cp["turn"], cp["path"]), []).append(cp)
        out = []
        session = self.store.get_session(session_id) or {}
        root = session.get("workspace", "")
        for (turn, path), cps in sorted(grouped.items()):
            before, after = cps[0]["before_content"], cps[-1]["after_content"]
            rel = _rel(root, path)
            diff, added, removed = unified_diff(rel, before, after or "")
            statuses = {c["status"] for c in cps}
            status = "rejected" if statuses == {"rejected"} else ("accepted" if statuses == {"accepted"} else "applied")
            out.append({
                "turn": turn, "path": rel, "abs_path": path, "diff": diff, "added": added, "removed": removed,
                "created": before is None, "status": status, "checkpoint_ids": [c["id"] for c in cps],
            })
        return out

    def accept_file(self, session_id: str, turn: int, path: str) -> dict[str, Any]:
        item = self._change(session_id, turn, path)
        self.store.set_checkpoint_status(item["checkpoint_ids"], "accepted")
        return {**item, "status": "accepted"}

    def reject_file(self, session_id: str, turn: int, path: str) -> dict[str, Any]:
        item = self._change(session_id, turn, path)
        if item["status"] == "rejected":
            return item
        cps = [c for c in self.store.checkpoints(session_id, turn) if c["id"] in item["checkpoint_ids"]]
        self._restore(Path(item["abs_path"]), cps[0]["before_content"], cps[-1]["after_content"])
        self.store.set_checkpoint_status(item["checkpoint_ids"], "rejected")
        self.hub.publish(session_id, turn, "notice", {"text": f"Reverted {item['path']}."})
        return {**item, "status": "rejected"}

    def undo_turn(self, session_id: str, turn: Optional[int] = None) -> list[str]:
        cps = self.store.checkpoints(session_id)
        if turn is None:
            turns = sorted({c["turn"] for c in cps if c["status"] != "rejected"})
            if not turns:
                return []
            turn = turns[-1]
        reverted = []
        for item in self.changes(session_id):
            if item["turn"] == turn and item["status"] != "rejected":
                self.reject_file(session_id, turn, item["path"])
                reverted.append(item["path"])
        return reverted

    def _change(self, session_id: str, turn: int, path: str) -> dict[str, Any]:
        wanted = path.replace("\\", "/")
        for item in self.changes(session_id):
            if item["turn"] == turn and (item["path"] == wanted or item["abs_path"] == path):
                return item
        raise AgentError(f"No agent change to {path} in turn {turn}.")

    @staticmethod
    def _restore(path: Path, before: Optional[str], expected_current: Optional[str]) -> None:
        current = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n") if path.is_file() else None
        if current != expected_current:
            raise AgentError(f"{path.name} was changed after the agent edited it; not reverting automatically.")
        if before is None:
            path.unlink(missing_ok=True)
        else:
            _write_text(path, before)

    # ============================================================ slash commands
    def _slash(self, session: dict[str, Any], text: str) -> dict[str, Any]:
        command, _, arg = text.partition(" ")
        command, arg = command.lower(), arg.strip()
        sid = session["id"]
        turn = self.store.last_turn(sid)
        if command == "/help":
            lines = "\n".join(f"- `{k}` — {v}" for k, v in SLASH_COMMANDS.items())
            self.hub.publish(sid, turn, "notice", {"text": "Commands:\n" + lines})
        elif command == "/clear":
            self.store.deactivate_messages(sid)
            self.store.update_session(sid, todos=[])
            self.hub.publish(sid, turn, "notice", {"text": "Context cleared. The model starts fresh (the transcript above is kept for reference).", "cleared": True})
            self.hub.publish(sid, turn, "todos", {"todos": []})
        elif command == "/compact":
            if not session.get("model_id") or not self.registry.adapter.is_loaded(session["model_id"]):
                raise AgentError("Load the session's model before compacting.")
            summary = self.compact(sid, turn=turn)
            if not summary:
                self.hub.publish(sid, turn, "notice", {"text": "Nothing to compact yet."})
        elif command == "/model":
            models = self.registry.list_models()
            if not arg:
                listing = "\n".join(
                    f"- {m.get('display_name') or m['name']}{' (loaded)' if m.get('loaded') else ''}" for m in models
                )
                self.hub.publish(sid, turn, "notice", {"text": "Models:\n" + listing})
            else:
                match = [m for m in models if arg.lower() in (m.get("display_name") or m["name"]).lower() or m["id"] == arg]
                if not match:
                    raise AgentError(f"No model matches '{arg}'.")
                chosen = sorted(match, key=lambda m: not m.get("loaded"))[0]
                self.store.update_session(sid, model_id=chosen["id"])
                note = "" if chosen.get("loaded") else " It is not loaded yet — load it in the Models panel."
                self.hub.publish(sid, turn, "notice", {"text": f"Model switched to {chosen.get('display_name') or chosen['name']}.{note}", "model_id": chosen["id"]})
        elif command == "/mode":
            if arg not in MODES:
                raise AgentError(f"Use /mode {'|'.join(MODES)}.")
            self.store.update_session(sid, mode=arg)
            self.hub.publish(sid, turn, "notice", {"text": f"Mode: {arg}.", "mode": arg})
        elif command == "/init":
            session = self.store.get_session(sid) or session
            if session["mode"] in ("ask", "plan"):
                raise AgentError("/init writes AGENTS.md, so it needs Agent or Auto mode.")
            return self._start_turn(session, self.prompts.render(self.prompts.get("init.md"), {}), display="/init")
        elif command == "/undo":
            reverted = self.undo_turn(sid)
            self.hub.publish(sid, turn, "notice", {"text": ("Reverted: " + ", ".join(reverted)) if reverted else "No agent edits to undo."})
        else:
            raise AgentError(f"Unknown command {command}. Type /help for the list.")
        return {"command": command, "started": False}


def _rel(root: str, path: str) -> str:
    try:
        return str(Path(path).relative_to(root)).replace("\\", "/")
    except ValueError:
        return path


def _friendly(exc: Exception) -> str:
    text = str(exc) or type(exc).__name__
    if "context" in text.lower() and ("exceed" in text.lower() or "too long" in text.lower()):
        return f"The conversation no longer fits in the model's context. Use /compact or /clear. ({text})"
    return text
