"""SQLite embedding cache keyed by (complaint_id, kind). kind = 'text' | 'image'."""
import sqlite3
import threading
from typing import Optional

import numpy as np

from app import config

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.CACHE_DB_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "complaint_id TEXT, kind TEXT, dim INTEGER, vec BLOB, "
            "PRIMARY KEY (complaint_id, kind))"
        )
        _conn.commit()
    return _conn


def get_embedding(complaint_id: str, kind: str) -> Optional[np.ndarray]:
    """Return cached float32 vector or None."""
    with _lock:
        row = _db().execute(
            "SELECT vec FROM embeddings WHERE complaint_id=? AND kind=?", (complaint_id, kind)
        ).fetchone()
    return np.frombuffer(row[0], dtype=np.float32) if row else None


def set_embedding(complaint_id: str, kind: str, vec: np.ndarray) -> None:
    """Store/replace a vector."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    with _lock:
        _db().execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
            (complaint_id, kind, v.size, v.tobytes()),
        )
        _db().commit()
