"""Persistence for coding-agent sessions (SQLite tables added in schema v4).

* ``agent_sessions``   — one row per session (workspace, model, mode, todos, options)
* ``agent_events``     — the UI transcript: every streamed event, replayable
* ``agent_messages``   — the model-facing conversation (OpenAI message dicts); rows are
                          deactivated (not deleted) by /clear and /compact
* ``agent_checkpoints``— file contents before/after each agent edit, for undo and
                          per-file accept/reject
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ...storage.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._seq_lock = threading.Lock()

    # ------------------------------------------------------------- sessions
    def create_session(
        self, workspace: str, model_id: Optional[str], mode: str = "agent", title: str = "New session"
    ) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        now = _now()
        self.db.execute(
            "INSERT INTO agent_sessions (id, workspace, title, model_id, mode, state, todos, options, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, 'idle', '[]', '{}', ?, ?)",
            (session_id, workspace, title, model_id, mode, now, now),
        )
        return self.get_session(session_id)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM agent_sessions WHERE id = ?", (session_id,))
        return self._session(row) if row else None

    def list_sessions(self, workspace: Optional[str] = None) -> list[dict[str, Any]]:
        if workspace:
            rows = self.db.query(
                "SELECT * FROM agent_sessions WHERE workspace = ? ORDER BY updated_at DESC", (workspace,)
            )
        else:
            rows = self.db.query("SELECT * FROM agent_sessions ORDER BY updated_at DESC")
        return [self._session(r) for r in rows]

    def update_session(self, session_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        allowed = {"title", "model_id", "mode", "state", "todos", "options", "workspace"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_session(session_id)
        for key in ("todos", "options"):
            if key in updates and not isinstance(updates[key], str):
                updates[key] = json.dumps(updates[key])
        sets = ", ".join(f"{k} = ?" for k in updates)
        self.db.execute(
            f"UPDATE agent_sessions SET {sets}, updated_at = ? WHERE id = ?",
            (*updates.values(), _now(), session_id),
        )
        return self.get_session(session_id)

    def touch(self, session_id: str) -> None:
        self.db.execute("UPDATE agent_sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    def delete_session(self, session_id: str) -> None:
        self.db.execute("DELETE FROM agent_sessions WHERE id = ?", (session_id,))

    @staticmethod
    def _session(row: Any) -> dict[str, Any]:
        data = dict(row)
        data["todos"] = json.loads(data.get("todos") or "[]")
        data["options"] = json.loads(data.get("options") or "{}")
        return data

    # --------------------------------------------------------------- events
    def append_event(self, session_id: str, turn: int, event_type: str, payload: dict[str, Any]) -> int:
        with self._seq_lock:
            row = self.db.query_one("SELECT COALESCE(MAX(seq), 0) AS s FROM agent_events WHERE session_id = ?", (session_id,))
            seq = int(row["s"]) + 1
            self.db.execute(
                "INSERT INTO agent_events (session_id, seq, turn, type, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, seq, turn, event_type, json.dumps(payload), _now()),
            )
        return seq

    def update_event(self, session_id: str, seq: int, payload: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE agent_events SET payload = ? WHERE session_id = ? AND seq = ?",
            (json.dumps(payload), session_id, seq),
        )

    def list_events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT seq, turn, type, payload, created_at FROM agent_events WHERE session_id = ? AND seq > ? ORDER BY seq",
            (session_id, after),
        )
        return [
            {"seq": r["seq"], "turn": r["turn"], "type": r["type"], "created_at": r["created_at"], **json.loads(r["payload"])}
            for r in rows
        ]

    def last_turn(self, session_id: str) -> int:
        row = self.db.query_one("SELECT COALESCE(MAX(turn), 0) AS t FROM agent_events WHERE session_id = ?", (session_id,))
        return int(row["t"])

    # ------------------------------------------------------------- messages
    def add_message(self, session_id: str, turn: int, message: dict[str, Any]) -> int:
        return self.db.execute(
            "INSERT INTO agent_messages (session_id, turn, message, active, created_at) VALUES (?, ?, ?, 1, ?)",
            (session_id, turn, json.dumps(message), _now()),
        )

    def deactivate_message(self, message_id: int) -> None:
        self.db.execute("UPDATE agent_messages SET active = 0 WHERE id = ?", (message_id,))

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, turn, message FROM agent_messages WHERE session_id = ? AND active = 1 ORDER BY id", (session_id,)
        )
        return [{"_id": r["id"], "_turn": r["turn"], **json.loads(r["message"])} for r in rows]

    def deactivate_messages(self, session_id: str, up_to_id: Optional[int] = None) -> None:
        if up_to_id is None:
            self.db.execute("UPDATE agent_messages SET active = 0 WHERE session_id = ?", (session_id,))
        else:
            self.db.execute(
                "UPDATE agent_messages SET active = 0 WHERE session_id = ? AND id <= ?", (session_id, up_to_id)
            )

    def deactivate_turn_messages(self, session_id: str, turn: int) -> None:
        self.db.execute("UPDATE agent_messages SET active = 0 WHERE session_id = ? AND turn >= ?", (session_id, turn))

    # ---------------------------------------------------------- checkpoints
    def add_checkpoint(
        self, session_id: str, turn: int, call_id: str, path: str, before: Optional[str], after: Optional[str]
    ) -> int:
        return self.db.execute(
            "INSERT INTO agent_checkpoints (session_id, turn, call_id, path, before_content, after_content, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 'applied', ?)",
            (session_id, turn, call_id, path, before, after, _now()),
        )

    def checkpoints(self, session_id: str, turn: Optional[int] = None) -> list[dict[str, Any]]:
        if turn is None:
            rows = self.db.query("SELECT * FROM agent_checkpoints WHERE session_id = ? ORDER BY id", (session_id,))
        else:
            rows = self.db.query(
                "SELECT * FROM agent_checkpoints WHERE session_id = ? AND turn = ? ORDER BY id", (session_id, turn)
            )
        return [dict(r) for r in rows]

    def set_checkpoint_status(self, checkpoint_ids: list[int], status: str) -> None:
        for cid in checkpoint_ids:
            self.db.execute("UPDATE agent_checkpoints SET status = ? WHERE id = ?", (status, cid))
