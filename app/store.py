"""Persistent run storage. Stores derived state only; the request's sensitive
object is never written here."""

import json
import os
import sqlite3


class Store:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(
            os.path.join(directory, "runs.sqlite3"), check_same_thread=False
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, run_id: str):
        row = self._conn.execute(
            "SELECT data FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, run_id: str, state: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, data) VALUES (?, ?)",
            (run_id, json.dumps(state, ensure_ascii=False)),
        )
        self._conn.commit()
