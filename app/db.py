"""Single SQLite store for: session memory, agent runs/steps, LLM cost ledger,
policy document registry, and the audit log."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    status TEXT,
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    input_ref TEXT,
    output_json TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON agent_runs(session_id);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
    seq INTEGER NOT NULL,
    step_name TEXT NOT NULL,
    prompt_name TEXT,
    prompt_version TEXT,
    input_json TEXT,
    output_json TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(run_id);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    run_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    metadata_json TEXT,
    ingested_by TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _dumps(obj: Any) -> str | None:
    return None if obj is None else json.dumps(obj, default=str)


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _row(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self._conn.close()

    # ---- audit -----------------------------------------------------------
    def audit(self, actor: str, action: str, *, resource_type: str | None = None,
              resource_id: str | None = None, status: str = "ok",
              request_id: str | None = None, details: dict | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO audit_log(ts,request_id,actor,action,resource_type,resource_id,status,details_json)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (utcnow(), request_id, actor, action, resource_type, resource_id, status, _dumps(details)),
            )

    def list_audit(self, limit: int = 100, action: str | None = None) -> list[dict[str, Any]]:
        if action:
            return self._rows("SELECT * FROM audit_log WHERE action=? ORDER BY id DESC LIMIT ?", (action, limit))
        return self._rows("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))

    # ---- runs / steps ----------------------------------------------------
    def create_run(self, session_id: str, agent_name: str, actor: str, input_ref: str | None) -> str:
        run_id = new_id("run")
        with self.tx() as c:
            c.execute(
                "INSERT INTO agent_runs(run_id,session_id,agent_name,actor,status,input_ref,started_at)"
                " VALUES(?,?,?,?,'running',?,?)",
                (run_id, session_id, agent_name, actor, input_ref, utcnow()),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, output: Any = None, error: str | None = None) -> None:
        totals = self._row(
            "SELECT COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(completion_tokens),0) c,"
            " COALESCE(SUM(cost_usd),0) cost FROM llm_calls WHERE run_id=?", (run_id,)) or {}
        with self.tx() as c:
            c.execute(
                "UPDATE agent_runs SET status=?, output_json=?, error=?, finished_at=?,"
                " total_prompt_tokens=?, total_completion_tokens=?, total_cost_usd=? WHERE run_id=?",
                (status, _dumps(output), error, utcnow(),
                 totals.get("p", 0), totals.get("c", 0), totals.get("cost", 0.0), run_id),
            )

    def add_step(self, run_id: str, seq: int, step_name: str, *, prompt_name: str | None = None,
                 prompt_version: str | None = None, input_data: Any = None, output_data: Any = None,
                 latency_ms: int | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO agent_steps(run_id,seq,step_name,prompt_name,prompt_version,input_json,output_json,"
                "latency_ms,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, seq, step_name, prompt_name, prompt_version, _dumps(input_data), _dumps(output_data),
                 latency_ms, utcnow()),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._row("SELECT * FROM agent_runs WHERE run_id=?", (run_id,))
        if run is None:
            return None
        run["output"] = json.loads(run.pop("output_json")) if run.get("output_json") else None
        steps = self._rows("SELECT * FROM agent_steps WHERE run_id=? ORDER BY seq", (run_id,))
        for s in steps:
            s["input"] = json.loads(s.pop("input_json")) if s.get("input_json") else None
            s["output"] = json.loads(s.pop("output_json")) if s.get("output_json") else None
        run["steps"] = steps
        run["llm_calls"] = self._rows("SELECT * FROM llm_calls WHERE run_id=? ORDER BY id", (run_id,))
        return run

    def list_runs(self, limit: int = 50, session_id: str | None = None) -> list[dict[str, Any]]:
        cols = ("run_id,session_id,agent_name,actor,status,input_ref,started_at,finished_at,"
                "total_prompt_tokens,total_completion_tokens,total_cost_usd")
        if session_id:
            return self._rows(f"SELECT {cols} FROM agent_runs WHERE session_id=? ORDER BY started_at DESC LIMIT ?",
                              (session_id, limit))
        return self._rows(f"SELECT {cols} FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,))

    # ---- cost ledger -----------------------------------------------------
    def record_llm_call(self, *, run_id: str | None, purpose: str, model: str, prompt_tokens: int,
                        completion_tokens: int, cost_usd: float, latency_ms: int) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO llm_calls(ts,run_id,purpose,model,prompt_tokens,completion_tokens,cost_usd,latency_ms)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (utcnow(), run_id, purpose, model, prompt_tokens, completion_tokens, cost_usd, latency_ms),
            )

    def cost_summary(self) -> dict[str, Any]:
        by_model = self._rows(
            "SELECT model, COUNT(*) calls, SUM(prompt_tokens) prompt_tokens, SUM(completion_tokens) completion_tokens,"
            " ROUND(SUM(cost_usd),6) cost_usd FROM llm_calls GROUP BY model ORDER BY cost_usd DESC")
        by_purpose = self._rows(
            "SELECT purpose, COUNT(*) calls, ROUND(SUM(cost_usd),6) cost_usd FROM llm_calls GROUP BY purpose")
        by_day = self._rows(
            "SELECT substr(ts,1,10) day, COUNT(*) calls, ROUND(SUM(cost_usd),6) cost_usd FROM llm_calls"
            " GROUP BY day ORDER BY day DESC LIMIT 30")
        total = self._row("SELECT COUNT(*) calls, ROUND(COALESCE(SUM(cost_usd),0),6) cost_usd FROM llm_calls") or {}
        return {"total": total, "by_model": by_model, "by_purpose": by_purpose, "by_day": by_day}

    # ---- memory ----------------------------------------------------------
    def add_memory(self, session_id: str, role: str, content: str, run_id: str | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO memory(session_id,run_id,role,content,created_at) VALUES(?,?,?,?,?)",
                      (session_id, run_id, role, content, utcnow()))

    def get_memory(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM memory WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit))
        return list(reversed(rows))

    def clear_memory(self, session_id: str) -> int:
        with self.tx() as c:
            return c.execute("DELETE FROM memory WHERE session_id=?", (session_id,)).rowcount

    # ---- document registry ----------------------------------------------
    def find_document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM documents WHERE sha256=?", (sha256,))

    def add_document(self, doc_id: str, filename: str, sha256: str, chunk_count: int,
                     embedding_model: str, metadata: dict, ingested_by: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO documents(doc_id,filename,sha256,chunk_count,embedding_model,metadata_json,"
                "ingested_by,ingested_at) VALUES(?,?,?,?,?,?,?,?)",
                (doc_id, filename, sha256, chunk_count, embedding_model, _dumps(metadata), ingested_by, utcnow()),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        docs = self._rows("SELECT * FROM documents ORDER BY ingested_at DESC")
        for d in docs:
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        return docs

    def delete_document(self, doc_id: str) -> bool:
        with self.tx() as c:
            return c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,)).rowcount > 0
