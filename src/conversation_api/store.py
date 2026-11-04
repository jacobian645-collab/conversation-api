"""Conversation storage.

`memory` is the default and needs nothing. `sqlite` keeps conversations across restarts (a Cloud Shell
session, a container restart with a mounted volume). For several replicas, put a shared store behind the
same interface (DynamoDB, Cosmos DB, Firestore, Redis); nothing else changes.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .schemas import Conversation


class ConversationStore(Protocol):
    kind: str

    async def get(self, conv_id: str) -> Conversation | None: ...
    async def put(self, conv: Conversation) -> None: ...
    async def delete(self, conv_id: str) -> bool: ...
    async def list(self, limit: int = 20) -> list[Conversation]: ...


class MemoryStore:
    kind = "memory"

    def __init__(self, max_items: int = 1000):
        self._items: OrderedDict[str, str] = OrderedDict()  # stored as JSON so callers never share objects
        self._max = max_items

    async def get(self, conv_id: str) -> Conversation | None:
        raw = self._items.get(conv_id)
        if raw is None:
            return None
        self._items.move_to_end(conv_id)
        return Conversation.model_validate_json(raw)

    async def put(self, conv: Conversation) -> None:
        self._items[conv.id] = conv.model_dump_json()
        self._items.move_to_end(conv.id)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    async def delete(self, conv_id: str) -> bool:
        return self._items.pop(conv_id, None) is not None

    async def list(self, limit: int = 20) -> list[Conversation]:
        ids = list(reversed(self._items))[:limit]
        return [Conversation.model_validate_json(self._items[i]) for i in ids]


class SqliteStore:
    kind = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                " id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS ix_updated ON conversations(updated_at)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:  # commit / rollback
                yield conn
        finally:
            conn.close()

    def _get(self, conv_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT data FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        return Conversation.model_validate_json(row[0]) if row else None

    def _put(self, conv: Conversation) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO conversations (id, data, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (conv.id, conv.model_dump_json(), conv.updated_at.isoformat()),
            )

    def _delete(self, conv_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,)).rowcount > 0

    def _list(self, limit: int) -> list[Conversation]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT data FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Conversation.model_validate_json(r[0]) for r in rows]

    async def get(self, conv_id: str) -> Conversation | None:
        return await asyncio.to_thread(self._get, conv_id)

    async def put(self, conv: Conversation) -> None:
        await asyncio.to_thread(self._put, conv)

    async def delete(self, conv_id: str) -> bool:
        return await asyncio.to_thread(self._delete, conv_id)

    async def list(self, limit: int = 20) -> list[Conversation]:
        return await asyncio.to_thread(self._list, limit)
