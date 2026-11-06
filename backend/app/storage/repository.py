"""Typed repository over the SQLite database (no ORM)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def _rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- conversations -------------------------------------------------
    def create_conversation(
        self,
        *,
        title: str = "New chat",
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        conversation_id = _new_id()
        now = _now()
        self.db.execute(
            "INSERT INTO conversations (id, title, model_id, system_prompt, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, title, model_id, system_prompt, now, now),
        )
        return self.get_conversation(conversation_id)  # type: ignore[return-value]

    def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        return _row(
            self.db.query_one(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            )
        )

    def list_conversations(
        self,
        *,
        search: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            clauses.append("archived = 0")
        if search:
            clauses.append("title LIKE ?")
            params.append(f"%{search}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM conversations {where} ORDER BY pinned DESC, updated_at DESC"
        return _rows(self.db.query(sql, params))

    def update_conversation(
        self, conversation_id: str, **fields: Any
    ) -> Optional[dict[str, Any]]:
        if not fields:
            return self.get_conversation(conversation_id)
        allowed = {"title", "model_id", "system_prompt", "pinned", "archived"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(
            f"UPDATE conversations SET {assignments} WHERE id = ?",
            (*fields.values(), conversation_id),
        )
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ---- messages ------------------------------------------------------
    def add_message(
        self, conversation_id: str, role: str, content: str, meta: Optional[dict] = None
    ) -> dict[str, Any]:
        now = _now()
        message_id = self.db.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps(meta) if meta else None,
                now,
            ),
        )
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "meta": meta,
            "created_at": now,
        }

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
        result = []
        for row in rows:
            item = dict(row)
            item["meta"] = json.loads(item["meta"]) if item.get("meta") else None
            result.append(item)
        return result

    # ---- registered models --------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        return _rows(self.db.query("SELECT * FROM registered_models ORDER BY name ASC"))

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        return _row(
            self.db.query_one("SELECT * FROM registered_models WHERE id = ?", (model_id,))
        )

    def get_model_by_path(self, path: str) -> Optional[dict[str, Any]]:
        return _row(
            self.db.query_one("SELECT * FROM registered_models WHERE path = ?", (path,))
        )

    def insert_model(self, model: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        self.db.execute(
            "INSERT INTO registered_models"
            " (id, name, display_name, description, path, format, quant, size_bytes, metadata, runtime, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                model["id"],
                model["name"],
                model.get("display_name"),
                model.get("description"),
                model["path"],
                model.get("format", "gguf"),
                model.get("quant"),
                model.get("size_bytes"),
                json.dumps(model.get("metadata")) if model.get("metadata") else None,
                model.get("runtime", "llama-cpp-python"),
                now,
                now,
            ),
        )
        return self.get_model(model["id"])  # type: ignore[return-value]

    def update_model(self, model_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        allowed = {"display_name", "description", "metadata"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        fields["updated_at"] = _now()
        if not fields:
            return self.get_model(model_id)
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(
            f"UPDATE registered_models SET {assignments} WHERE id = ?",
            (*fields.values(), model_id),
        )
        return self.get_model(model_id)

    def delete_model(self, model_id: str) -> None:
        self.db.execute("DELETE FROM registered_models WHERE id = ?", (model_id,))

    def refresh_model(self, path: str, item: dict[str, Any]) -> None:
        """Update discoverable metadata for an already-registered model file."""
        self.db.execute(
            "UPDATE registered_models SET name = ?, quant = ?, size_bytes = ?,"
            " format = ?, metadata = ?, updated_at = ? WHERE path = ?",
            (
                item.get("name") or item.get("path"),
                item.get("quant"),
                item.get("size_bytes"),
                item.get("format", "gguf"),
                json.dumps(item),
                _now(),
                path,
            ),
        )

    # ---- settings ------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        serialized = json.dumps(value)
        self.db.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, serialized),
        )

    # ---- tasks and tool logs ------------------------------------------
    def create_task(self, task_id: str, kind: str, payload: Optional[dict] = None) -> None:
        now = _now()
        self.db.execute(
            "INSERT INTO tasks (id, kind, status, payload, created_at, updated_at)"
            " VALUES (?, ?, 'pending', ?, ?, ?)",
            (task_id, kind, json.dumps(payload) if payload else None, now, now),
        )

    def update_task(
        self, task_id: str, status: str, result: Optional[dict] = None
    ) -> None:
        self.db.execute(
            "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (status, json.dumps(result) if result else None, _now(), task_id),
        )

    def add_tool_log(
        self,
        *,
        tool_name: str,
        status: str,
        permission_level: Optional[str] = None,
        target: Optional[str] = None,
        result_summary: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO tool_logs (task_id, tool_name, permission_level, target, status, result_summary, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, tool_name, permission_level, target, status, result_summary, _now()),
        )

    def list_tool_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        return _rows(
            self.db.query(
                "SELECT * FROM tool_logs ORDER BY id DESC LIMIT ?", (limit,)
            )
        )

    # ---- approvals -----------------------------------------------------
    def create_approval(
        self,
        *,
        approval_id: str,
        tool_name: str,
        permission_level: str,
        args: dict[str, Any],
        summary: str,
    ) -> None:
        self.db.execute(
            "INSERT INTO approvals (id, tool_name, permission_level, args, summary, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (approval_id, tool_name, permission_level, json.dumps(args), summary, _now()),
        )

    def get_approval(self, approval_id: str) -> Optional[dict[str, Any]]:
        row = self.db.query_one("SELECT * FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            return None
        item = dict(row)
        item["args"] = json.loads(item["args"]) if item.get("args") else {}
        return item

    def list_approvals(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        if status:
            rows = self.db.query(
                "SELECT * FROM approvals WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        else:
            rows = self.db.query("SELECT * FROM approvals ORDER BY created_at DESC")
        result = []
        for row in rows:
            item = dict(row)
            item["args"] = json.loads(item["args"]) if item.get("args") else {}
            result.append(item)
        return result

    def decide_approval(self, approval_id: str, approved: bool) -> None:
        self.db.execute(
            "UPDATE approvals SET status = ?, decided_at = ? WHERE id = ?",
            ("approved" if approved else "denied", _now(), approval_id),
        )

    # ---- retrieval (RAG) ----------------------------------------------
    def add_rag_source(self, source_id: str, name: str, kind: str, path: Optional[str]) -> None:
        self.db.execute(
            "INSERT INTO rag_sources (id, name, kind, path, created_at) VALUES (?, ?, ?, ?, ?)",
            (source_id, name, kind, path, _now()),
        )

    def add_rag_chunks(self, source_id: str, chunks: list[str]) -> None:
        rows = [(source_id, i, text) for i, text in enumerate(chunks)]
        self.db.executemany(
            "INSERT INTO rag_chunks (source_id, chunk_index, text) VALUES (?, ?, ?)", rows
        )

    def list_rag_sources(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT s.*, COUNT(c.id) AS chunk_count FROM rag_sources s"
            " LEFT JOIN rag_chunks c ON c.source_id = s.id"
            " GROUP BY s.id ORDER BY s.created_at DESC"
        )
        return _rows(rows)

    def list_rag_chunks(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT c.id, c.source_id, c.text, s.name AS source_name"
            " FROM rag_chunks c JOIN rag_sources s ON s.id = c.source_id"
        )
        return _rows(rows)

    def delete_rag_source(self, source_id: str) -> None:
        self.db.execute("DELETE FROM rag_sources WHERE id = ?", (source_id,))


def _new_id() -> str:
    import uuid

    return uuid.uuid4().hex
