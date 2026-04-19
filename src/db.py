import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "applied-ai-take-home-database" / "synthetic_startup.sqlite"

_local = threading.local()


def _new_conn() -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


class _ConnProxy:
    """Thread-local sqlite connection. Attribute access (execute, row_factory, etc.)
    transparently routes to a per-thread connection so concurrent subagents don't collide."""

    def _conn(self) -> sqlite3.Connection:
        c = getattr(_local, "conn", None)
        if c is None:
            c = _new_conn()
            _local.conn = c
        return c

    def __getattr__(self, name):
        return getattr(self._conn(), name)


conn = _ConnProxy()


def resolve_customer(name_or_id: str):
    """Return (customer_id, canonical_name) for an id, exact name, or single fuzzy match, else None."""
    name_or_id = name_or_id.strip()
    if not name_or_id:
        return None
    row = conn.execute(
        "SELECT customer_id, name FROM customers WHERE customer_id = ? OR lower(name) = lower(?)",
        (name_or_id, name_or_id),
    ).fetchone()
    if row:
        return row["customer_id"], row["name"]
    rows = conn.execute(
        "SELECT customer_id, name FROM customers WHERE lower(name) LIKE lower(?) ORDER BY name",
        (f"%{name_or_id}%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]["customer_id"], rows[0]["name"]
    return None


def customer_suggestions(name: str, limit: int = 5) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM customers WHERE lower(name) LIKE lower(?) ORDER BY name LIMIT ?",
        (f"%{name}%", limit),
    ).fetchall()
    return [r["name"] for r in rows]
