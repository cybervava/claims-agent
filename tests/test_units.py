import json

import pytest

from app.db import Database
from app.ingest import UnsupportedFileError, chunk_text, extract_text
from app.pricing import estimate_cost_usd, resolve_model_key
from app.prompts import PromptStore


# ---- pricing ---------------------------------------------------------------
def test_pricing_resolves_versioned_azure_model_names():
    assert resolve_model_key("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert resolve_model_key("gpt-4o-2024-11-20") == "gpt-4o"
    assert resolve_model_key("unknown-model") is None


def test_pricing_cost_math():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost_usd("text-embedding-3-small", 1_000_000, 0) == pytest.approx(0.02)
    assert estimate_cost_usd("nope", 100, 100) == 0.0


def test_pricing_override_env(monkeypatch):
    monkeypatch.setenv("PRICING_JSON", json.dumps({"gpt-4o-mini": {"input": 1.0, "output": 1.0}}))
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(1.0)


# ---- prompts ---------------------------------------------------------------
def test_prompt_store_renders_and_versions(tmp_path):
    (tmp_path / "hello.md").write_text('Hi ${name}, JSON stays {"a": 1}', encoding="utf-8")
    ps = PromptStore(tmp_path)
    r = ps.render("hello", name="Bob")
    assert r.text == 'Hi Bob, JSON stays {"a": 1}'
    assert len(r.version) == 12
    assert ps.list() == [{"name": "hello", "version": r.version}]
    with pytest.raises(FileNotFoundError):
        ps.render("missing")


def test_real_prompts_exist(prompts):
    names = {p["name"] for p in prompts.list()}
    assert {"claim_extractor", "claim_validator", "claim_validator_user"} <= names
    rendered = prompts.render("claim_validator_user", memory="M", claim_json="{}", clauses="C")
    assert "${" not in rendered.text


# ---- ingest ----------------------------------------------------------------
def test_chunk_text_respects_size_and_overlap():
    paras = [f"Paragraph {i} " + "x" * 80 for i in range(10)]
    chunks = chunk_text("\n\n".join(paras), chunk_size=200, overlap=30)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1
    assert chunks[1].startswith(chunks[0][-30:].split("\n\n")[-1][:5])


def test_chunk_text_hard_splits_oversized_paragraph():
    chunks = chunk_text("y" * 1000, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(c[50:] if i else c for i, c in enumerate(chunks)).startswith("y" * 300)


def test_chunk_text_validates_args():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)


def test_extract_text_formats():
    assert extract_text("a.txt", b"hello") == "hello"
    assert '"k": 1' in extract_text("a.json", b'{"k":1}')
    with pytest.raises(UnsupportedFileError):
        extract_text("a.exe", b"x")
    with pytest.raises(UnsupportedFileError):
        extract_text("a.json", b"{not json")


def test_extract_text_docx_and_pdf():
    import io
    import docx
    from pypdf import PdfWriter

    d = docx.Document()
    d.add_paragraph("clause one")
    buf = io.BytesIO()
    d.save(buf)
    assert extract_text("p.docx", buf.getvalue()) == "clause one"

    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    pbuf = io.BytesIO()
    w.write(pbuf)
    assert extract_text("p.pdf", pbuf.getvalue()) == ""


# ---- db --------------------------------------------------------------------
def test_db_run_lifecycle_and_cost_rollup(db: Database):
    run_id = db.create_run("s1", "agent", "alice", "claim.txt")
    db.record_llm_call(run_id=run_id, purpose="a", model="gpt-4o-mini", prompt_tokens=100,
                       completion_tokens=50, cost_usd=0.001, latency_ms=5)
    db.record_llm_call(run_id=run_id, purpose="b", model="text-embedding-3-small", prompt_tokens=10,
                       completion_tokens=0, cost_usd=0.0001, latency_ms=2)
    db.add_step(run_id, 1, "extract", prompt_name="p", prompt_version="v", input_data={"a": 1}, output_data=[1])
    db.finish_run(run_id, "completed", output={"decision": "APPROVE"})
    run = db.get_run(run_id)
    assert run["status"] == "completed"
    assert run["total_prompt_tokens"] == 110 and run["total_completion_tokens"] == 50
    assert run["total_cost_usd"] == pytest.approx(0.0011)
    assert run["steps"][0]["input"] == {"a": 1} and run["output"] == {"decision": "APPROVE"}
    assert len(run["llm_calls"]) == 2
    assert db.list_runs(session_id="s1")[0]["run_id"] == run_id
    assert db.get_run("nope") is None
    summary = db.cost_summary()
    assert summary["total"]["calls"] == 2
    assert {m["model"] for m in summary["by_model"]} == {"gpt-4o-mini", "text-embedding-3-small"}


def test_db_memory_audit_documents(db: Database):
    db.add_memory("s", "claim", "first")
    db.add_memory("s", "decision", "second")
    assert [m["content"] for m in db.get_memory("s")] == ["first", "second"]
    assert [m["content"] for m in db.get_memory("s", limit=1)] == ["second"]
    assert db.clear_memory("s") == 2 and db.get_memory("s") == []

    db.audit("bob", "x.y", resource_type="t", resource_id="1", details={"k": "v"})
    rows = db.list_audit(action="x.y")
    assert rows[0]["actor"] == "bob" and json.loads(rows[0]["details_json"]) == {"k": "v"}

    db.add_document("doc_1", "f.md", "sha", 3, "emb", {"product": "motor"}, "bob")
    assert db.find_document_by_hash("sha")["doc_id"] == "doc_1"
    assert db.list_documents()[0]["metadata"] == {"product": "motor"}
    assert db.delete_document("doc_1") is True and db.delete_document("doc_1") is False
