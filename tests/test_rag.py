"""RAG path: Azure AI Search client (mock transport), service (fake backends), API endpoints."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMError
from app.main import create_app
from app.pricing import estimate_cost_usd
from app.rag import NO_CONTEXT, RagService, build_context, to_source
from app.search import AzureSearchClient, SearchDocument, SearchError

DOCS = [
    SearchDocument(id="d1", title="gadget_electronics_policy.pdf", score=2.5,
                   content="Section 4 — Excess\n4.2 An additional excess of 50 applies to every loss claim.",
                   highlights=["4.2 An additional excess of 50 applies"]),
    SearchDocument(id="d2", title="pet_health_policy.pdf", score=1.1, content="x" * 50, highlights=[]),
]


class FakeSearch:
    def __init__(self, docs=DOCS, error: Exception | None = None):
        self.docs, self.error, self.queries = docs, error, []

    def search(self, query, *, top_k):
        self.queries.append((query, top_k))
        if self.error:
            raise self.error
        return self.docs[:top_k]


class FakeChat:
    def __init__(self, db, answer="The loss excess is 50 [gadget_electronics_policy.pdf].", error=None):
        self.db, self.answer, self.error, self.messages = db, answer, error, []

    def complete(self, messages, *, run_id, purpose):
        self.messages.append(messages)
        if self.error:
            raise self.error
        self.db.record_llm_call(run_id=run_id, purpose=purpose, model="gpt-5-mini-2025-08-07", prompt_tokens=800,
                                completion_tokens=60, cost_usd=estimate_cost_usd("gpt-5-mini", 800, 60), latency_ms=5)
        return self.answer, {"model": "gpt-5-mini-2025-08-07", "prompt_tokens": 800, "completion_tokens": 60,
                             "cost_usd": estimate_cost_usd("gpt-5-mini", 800, 60), "latency_ms": 5}


@pytest.fixture
def rag(settings, prompts, db):
    return RagService(settings, FakeSearch(), FakeChat(db), prompts, db)


# ---- pure helpers -----------------------------------------------------------
def test_build_context_labels_and_truncates():
    ctx = build_context(DOCS, max_chars_per_doc=20)
    assert ctx.startswith("[gadget_electronics_policy.pdf]\nSection 4 — Excess\n4")
    assert "[... truncated ...]" in ctx and "[pet_health_policy.pdf]" in ctx
    assert build_context([], 100) == NO_CONTEXT


def test_to_source_prefers_highlight():
    assert to_source(DOCS[0]).snippet == "4.2 An additional excess of 50 applies"
    assert to_source(DOCS[1]).snippet == "x" * 50


def test_gpt5_mini_priced():
    assert estimate_cost_usd("gpt-5-mini-2025-08-07", 1_000_000, 0) == 0.25


# ---- Azure AI Search client --------------------------------------------------
def _client(handler):
    return AzureSearchClient("https://s.search.windows.net", "idx", "key",
                             transport=httpx.MockTransport(handler))


def test_search_client_builds_request_and_parses_hits():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"value": [
            {"@search.score": 1.5, "id": "a", "title": "t.pdf", "content": " body ",
             "@search.highlights": {"content": [" hit ", ""]}},
            {"@search.score": None, "id": None, "title": None, "content": None},
        ]})

    hits = _client(handler).search("theft excess", top_k=2)
    assert seen["url"].startswith("https://s.search.windows.net/indexes/idx/docs/search?api-version=")
    assert seen["key"] == "key" and seen["body"]["search"] == "theft excess" and seen["body"]["top"] == 2
    assert hits[0] == SearchDocument(id="a", title="t.pdf", score=1.5, content="body", highlights=["hit"])
    assert hits[1].title == "untitled" and hits[1].score == 0.0 and hits[1].content == ""


def test_search_client_wraps_http_errors():
    with pytest.raises(SearchError, match="403"):
        _client(lambda r: httpx.Response(403, text="forbidden")).search("q")

    def boom(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(SearchError, match="no route"):
        _client(boom).search("q")


def test_search_client_requires_config():
    with pytest.raises(ValueError):
        AzureSearchClient("", "idx", "key")


# ---- service ---------------------------------------------------------------
def test_rag_ask_records_run_steps_cost_and_audit(rag, db):
    resp = rag.ask("What is the loss excess?", top_k=1, actor="alice", request_id="r1")
    assert resp.answer.startswith("The loss excess is 50")
    assert [s.title for s in resp.sources] == ["gadget_electronics_policy.pdf"]
    assert resp.model.startswith("gpt-5-mini") and resp.cost_usd > 0 and resp.prompt_tokens == 800
    assert rag.search.queries == [("What is the loss excess?", 1)]

    user_msg = rag.chat.messages[0][1]["content"]
    assert "[gadget_electronics_policy.pdf]" in user_msg and "Question: What is the loss excess?" in user_msg
    assert "pet_health" not in user_msg

    run = db.get_run(resp.run_id)
    assert run["status"] == "completed" and run["agent_name"] == "policy-rag"
    assert [s["step_name"] for s in run["steps"]] == ["retrieve", "answer"]
    assert run["total_cost_usd"] == pytest.approx(resp.cost_usd)
    audit = db.list_audit(action="rag.ask")
    assert audit[0]["actor"] == "alice" and audit[0]["request_id"] == "r1"


def test_rag_ask_with_no_hits_sends_no_context_marker(settings, prompts, db):
    rag = RagService(settings, FakeSearch(docs=[]), FakeChat(db, answer="Not in the documents."), prompts, db)
    resp = rag.ask("anything")
    assert resp.sources == [] and NO_CONTEXT in rag.chat.messages[0][1]["content"]


def test_rag_ask_failure_marks_run_failed(settings, prompts, db):
    rag = RagService(settings, FakeSearch(error=SearchError("down")), FakeChat(db), prompts, db)
    with pytest.raises(SearchError):
        rag.ask("q", actor="bob")
    run = db.get_run(db.list_runs()[0]["run_id"])
    assert run["status"] == "failed" and "down" in run["error"]
    assert db.list_audit(action="rag.ask")[0]["status"] == "error"


# ---- API -------------------------------------------------------------------
def test_rag_endpoints_503_when_unconfigured(client):
    assert client.get("/health").json()["rag"]["enabled"] is False
    r = client.post("/rag/ask", json={"question": "q"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]
    assert client.post("/rag/search", json={"question": "q"}).status_code == 503


@pytest.fixture
def rag_client(settings, monkeypatch, tmp_path):
    import app.deps as deps
    from tests.conftest import FakeLLM

    state = {}

    def fake_build_rag(s, prompts, db):
        state["rag"] = RagService(s, FakeSearch(), FakeChat(db), prompts, db)
        return state["rag"]

    monkeypatch.setattr(deps, "LLMClient", lambda s, d: FakeLLM(s, d))
    monkeypatch.setattr(deps, "build_rag", fake_build_rag)
    with TestClient(create_app(settings)) as c:
        yield c, state


def test_rag_ask_endpoint(rag_client):
    c, state = rag_client
    r = c.post("/rag/ask", json={"question": "loss excess?", "top_k": 2}, headers={"x-actor": "web"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("The loss excess") and len(body["sources"]) == 2
    assert body["sources"][0]["snippet"] == "4.2 An additional excess of 50 applies"
    assert c.get(f"/claims/runs/{body['run_id']}").json()["agent_name"] == "policy-rag"
    assert c.get("/costs").json()["total"]["cost_usd"] > 0

    hits = c.post("/rag/search", json={"question": "excess"}).json()
    assert [h["id"] for h in hits] == ["d1", "d2"] and "content" in hits[0]


def test_rag_ask_endpoint_validates_and_maps_errors(rag_client):
    c, state = rag_client
    assert c.post("/rag/ask", json={"question": ""}).status_code == 422
    assert c.post("/rag/ask", json={"question": "q", "top_k": 0}).status_code == 422

    state["rag"].chat.error = LLMError("empty")
    r = c.post("/rag/ask", json={"question": "q"})
    assert r.status_code == 502 and "LLM error" in r.json()["detail"]

    state["rag"].chat.error = None
    state["rag"].search.error = SearchError("index missing")
    r = c.post("/rag/ask", json={"question": "q"})
    assert r.status_code == 502 and "index missing" in r.json()["detail"]
    assert c.post("/rag/search", json={"question": "q"}).status_code == 502


def test_rag_settings_enabled_flag():
    base = dict(_env_file=None, azure_openai_endpoint="https://x", azure_openai_api_key="k")
    assert Settings(**base).rag_enabled is False
    assert Settings(**base, azure_search_endpoint="https://s", azure_search_api_key="k", azure_search_index="i",
                    rag_chat_base_url="https://o/openai/v1").rag_enabled is True
