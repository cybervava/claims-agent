import io
import json
from pathlib import Path

import pytest

from app.llm import LLMError

POLICY = Path("sample_data/policies/motor_comprehensive_policy.md").read_text()


# ---- vector store ------------------------------------------------------------
def test_store_ingest_search_delete(store, db):
    res = store.ingest("motor.md", POLICY.encode(), {"product": "motor"}, "alice")
    assert res.chunk_count > 1 and not res.skipped_duplicate
    dup = store.ingest("motor.md", POLICY.encode(), {}, "alice")
    assert dup.skipped_duplicate and dup.doc_id == res.doc_id
    assert store.stats()["chunks"] == res.chunk_count

    hits = store.search("theft excess police report", top_k=3)
    assert len(hits) == 3 and all(h.doc_id == res.doc_id for h in hits)
    assert hits[0].score >= hits[-1].score
    assert store.search("x", where={"product": "home"}) == []

    assert store.delete(res.doc_id) is True
    assert store.stats()["chunks"] == 0 and store.search("anything") == []


def test_store_rejects_empty(store):
    with pytest.raises(ValueError):
        store.ingest("empty.txt", b"   ", {}, "a")


# ---- agent -----------------------------------------------------------------
def test_agent_run_persists_steps_memory_cost_audit(agent, store, db, llm):
    store.ingest("motor.md", POLICY.encode(), {}, "alice")
    resp = agent.run("some claim text", filename="c.txt", session_id="sess-1", actor="alice", request_id="r1")
    assert resp.validation.decision == "APPROVE"
    assert resp.claim.policy_number == "MC-1"
    assert len(resp.evidence) <= 4 and resp.cost_usd > 0
    run = db.get_run(resp.run_id)
    assert [s["step_name"] for s in run["steps"]] == ["extract", "retrieve", "validate"]
    assert run["steps"][0]["prompt_version"] == agent.prompts.version("claim_extractor")
    assert run["status"] == "completed" and run["total_cost_usd"] == pytest.approx(resp.cost_usd)
    purposes = {c["purpose"] for c in run["llm_calls"]}
    assert {"extract_claim", "embed_query", "validate_claim"} <= purposes
    mem = db.get_memory("sess-1")
    assert [m["role"] for m in mem] == ["claim", "decision"]
    actions = [a["action"] for a in db.list_audit()]
    assert "claim.validate.start" in actions and "claim.validate.completed" in actions


def test_agent_memory_feeds_next_validation(agent, store, llm):
    store.ingest("motor.md", POLICY.encode(), {}, "a")
    agent.run("first", filename="c.txt", session_id="s", actor="a")
    agent.run("second", filename="c.txt", session_id="s", actor="a")
    validate_calls = [c for c in llm.calls if c["purpose"] == "validate_claim"]
    assert "(none)" in validate_calls[0]["messages"][1]["content"]
    assert "APPROVE (confidence 0.90)" in validate_calls[1]["messages"][1]["content"]


def test_agent_failure_marks_run_failed(agent, db, llm, monkeypatch):
    def boom(*a, **k):
        raise LLMError("refused")

    monkeypatch.setattr(llm, "chat_structured", boom)
    with pytest.raises(LLMError):
        agent.run("x", filename="c.txt", session_id="s", actor="a")
    run = db.list_runs()[0]
    assert run["status"] == "failed"
    assert db.list_audit(action="claim.validate.failed")


# ---- API -------------------------------------------------------------------
def test_api_end_to_end(client):
    assert client.get("/health").json()["status"] == "ok"

    r = client.post("/policies/ingest", files={"file": ("motor.md", POLICY.encode(), "text/markdown")},
                    data={"metadata": json.dumps({"product": "motor"})}, headers={"x-actor": "alice"})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]

    r = client.post("/policies/ingest-text", json={"filename": "n.txt", "text": "Notice period 30 days."})
    assert r.status_code == 200
    assert len(client.get("/policies").json()["documents"]) == 2

    hits = client.post("/policies/search", json={"query": "theft excess", "top_k": 2}).json()
    assert len(hits) == 2 and "chunk_id" in hits[0]

    r = client.post("/claims/validate", files={"file": ("claim.txt", b"rear ended", "text/plain")},
                    data={"session_id": "s9"}, headers={"x-actor": "alice"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["validation"]["decision"] == "APPROVE" and body["session_id"] == "s9"
    assert "x-request-id" in r.headers

    r = client.post("/claims/validate-text", json={"claim_text": "again", "session_id": "s9"})
    assert r.status_code == 200
    assert len(client.get("/memory/s9").json()) == 4
    assert len(client.get("/claims/runs", params={"session_id": "s9"}).json()) == 2
    assert client.get(f"/claims/runs/{body['run_id']}").json()["steps"][2]["step_name"] == "validate"
    assert client.get("/claims/runs/nope").status_code == 404

    costs = client.get("/costs").json()
    assert costs["total"]["calls"] > 0 and costs["by_model"]
    audit = client.get("/audit", params={"action": "http.request"}).json()
    assert any(a["resource_id"] == "POST /claims/validate" and a["actor"] == "alice" for a in audit)
    assert {p["name"] for p in client.get("/prompts").json()} >= {"claim_extractor"}

    assert client.delete("/memory/s9").json()["deleted"] == 4
    assert client.delete(f"/policies/{doc_id}").status_code == 200
    assert client.delete(f"/policies/{doc_id}").status_code == 404


def test_api_validation_errors(client):
    assert client.post("/policies/ingest", files={"file": ("x.exe", b"bin")}).status_code == 400
    assert client.post("/policies/ingest", files={"file": ("x.txt", b"ok")}, data={"metadata": "[1]"}).status_code == 400
    assert client.post("/policies/ingest", files={"file": ("x.txt", b"ok")}, data={"metadata": "{bad"}).status_code == 400
    assert client.post("/claims/validate", files={"file": ("c.txt", b"   ")}).status_code == 400
    assert client.post("/claims/validate", files={"file": ("c.bin", b"x")}).status_code == 400
    assert client.post("/claims/validate-text", json={"claim_text": ""}).status_code == 400


def test_api_upload_size_limit(client):
    client.app.state.ctx.settings.max_upload_bytes = 10
    assert client.post("/policies/ingest", files={"file": ("x.txt", b"x" * 11)}).status_code == 413
    assert client.post("/claims/validate", files={"file": ("x.txt", b"x" * 11)}).status_code == 413


def test_api_llm_error_maps_to_502(client, monkeypatch):
    def boom(*a, **k):
        raise LLMError("refused")

    monkeypatch.setattr(client.app.state.ctx.llm, "chat_structured", boom)
    r = client.post("/claims/validate-text", json={"claim_text": "x"})
    assert r.status_code == 502


def test_kb_browser_and_chunks_endpoint(client):
    r = client.get("/kb")
    assert r.status_code == 200 and "Policy KB Browser" in r.text
    assert "text/html" in r.headers["content-type"]
    doc_id = client.post("/policies/ingest-text", json={"filename": "m.md", "text": POLICY}).json()["doc_id"]
    chunks = client.get(f"/policies/{doc_id}/chunks").json()
    assert len(chunks) > 1
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["chunk_id"] == f"{doc_id}:0"
    assert "embedding_dims" not in chunks[0]
    with_vec = client.get(f"/policies/{doc_id}/chunks", params={"include_embeddings": "true"}).json()
    assert with_vec[0]["embedding_dims"] == 32 and len(with_vec[0]["embedding_preview"]) == 8
    assert client.get("/policies/doc_missing/chunks").status_code == 404
