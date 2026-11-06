"""SQLite persistence with an explicit, idempotent schema-version migration layer.

The schema version is tracked with ``PRAGMA user_version``. Migrations are plain
SQL scripts applied in order; each is wrapped in a transaction so a failed
migration never leaves the database half-migrated. Connections are short-lived
(one per operation) which is safe under FastAPI's threadpool and simple to reason
about; WAL mode keeps concurrent reads/writes fast and crash-resilient.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# Each entry is a complete migration script applied atomically.
MIGRATIONS: list[str] = [
    # v1: conversations, messages, model registry, settings, tasks, tool logs.
    """
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New chat',
        model_id TEXT,
        system_prompt TEXT,
        pinned INTEGER NOT NULL DEFAULT 0,
        archived INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        meta TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);

    CREATE TABLE registered_models (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        display_name TEXT,
        description TEXT,
        path TEXT NOT NULL UNIQUE,
        format TEXT NOT NULL DEFAULT 'gguf',
        quant TEXT,
        size_bytes INTEGER,
        metadata TEXT,
        runtime TEXT NOT NULL DEFAULT 'llama-cpp-python',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        conversation_id TEXT,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        payload TEXT,
        result TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE tool_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        tool_name TEXT NOT NULL,
        permission_level TEXT,
        target TEXT,
        status TEXT NOT NULL,
        result_summary TEXT,
        created_at TEXT NOT NULL
    );
    """,
    # v2: approval requests (pending/approved/denied) for the permission system.
    """
    CREATE TABLE approvals (
        id TEXT PRIMARY KEY,
        tool_name TEXT NOT NULL,
        permission_level TEXT NOT NULL,
        args TEXT NOT NULL,
        summary TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        decided_at TEXT
    );
    """,
    # v3: local retrieval (RAG) sources and chunks.
    """
    CREATE TABLE rag_sources (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        path TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE rag_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL REFERENCES rag_sources(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL
    );
    CREATE INDEX idx_rag_chunks_source ON rag_chunks(source_id);
    """,
    # v4: coding-agent sessions, their event transcript, model-facing messages, and
    # file checkpoints (undo / per-file accept-reject of agent edits).
    """
    CREATE TABLE agent_sessions (
        id TEXT PRIMARY KEY,
        workspace TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT 'New session',
        model_id TEXT,
        mode TEXT NOT NULL DEFAULT 'agent',
        state TEXT NOT NULL DEFAULT 'idle',
        todos TEXT,
        options TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX idx_agent_sessions_workspace ON agent_sessions(workspace, updated_at);

    CREATE TABLE agent_events (
        session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        turn INTEGER NOT NULL DEFAULT 0,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, seq)
    );

    CREATE TABLE agent_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
        turn INTEGER NOT NULL DEFAULT 0,
        message TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_agent_messages_session ON agent_messages(session_id, active, id);

    CREATE TABLE agent_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES agent_sessions(id) ON DELETE CASCADE,
        turn INTEGER NOT NULL,
        call_id TEXT,
        path TEXT NOT NULL,
        before_content TEXT,
        after_content TEXT,
        status TEXT NOT NULL DEFAULT 'applied',
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_agent_checkpoints_session ON agent_checkpoints(session_id, turn);
    """,
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for index, script in enumerate(MIGRATIONS[version:], start=version + 1):
                with conn:  # transaction per migration
                    conn.executescript(script)
                    conn.execute(f"PRAGMA user_version = {index}")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.connect() as conn:
            cursor = conn.execute(sql, tuple(params))
            return cursor.lastrowid or 0

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, tuple(params)).fetchone()

    def transaction(self, statements: Iterable[tuple[str, Sequence[Any]]]) -> None:
        with self.connect() as conn:
            with conn:
                for sql, params in statements:
                    conn.execute(sql, tuple(params))
