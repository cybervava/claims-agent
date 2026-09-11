"""Shared fixtures: fake LLM (deterministic, offline) + real SQLite/ChromaDB in a temp dir."""
import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from app.agent import ClaimAgent
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.pricing import estimate_cost_usd
from app.prompts import PromptStore
from app.schemas import ClaimItem, ClaimRecord, Finding, ValidationResult
from app.vectorstore import PolicyStore


def fake_vector(text: str, dims: int = 32) -> list[float]:
    """Bag-of-words hashed embedding so similar texts land close together."""
    vec = [0.0] * dims
    for tok in text.lower().split():
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeLLM:
    """Stands in for LLMClient. Records calls to the same cost ledger the real client uses."""

    def __init__(self, settings: Settings, db: Database, decision: str = "APPROVE"):
        self.settings, self.db, self.decision = settings, db, decision
        self.calls: list[dict] = []

    def _meter(self, run_id, purpose, model, p, c):
        self.db.record_llm_call(run_id=run_id, purpose=purpose, model=model, prompt_tokens=p,
                                completion_tokens=c, cost_usd=estimate_cost_usd(model, p, c), latency_ms=1)

    def chat_structured(self, messages, schema, *, run_id, purpose):
        self.calls.append({"purpose": purpose, "messages": messages, "schema": schema.__name__})
        self._meter(run_id, purpose, "gpt-4o-mini-2024-07-18", 1000, 200)
        if schema is ClaimRecord:
            return ClaimRecord(policy_number="MC-1", claimant_name="Test", claim_type="motor collision",
                               incident_date="2026-08-01", reported_date="2026-08-02", location="X",
                               total_claimed_amount=1500.0, currency="USD", description="rear-ended at lights",
                               items=[ClaimItem(description="bumper repair", amount=1500.0)],
                               supporting_documents=["photos"], red_flags=[]), {}
        if schema is ValidationResult:
            return ValidationResult(decision=self.decision, confidence=0.9, summary="ok",
                                    findings=[Finding(rule="covered peril", status="PASS",
                                                      evidence_chunk_ids=[], reasoning="collision covered")],
                                    covered_amount_estimate=1000.0, missing_information=[],
                                    recommended_next_steps=[]), {}
        raise AssertionError(f"unexpected schema {schema}")

    def embed(self, texts, *, run_id=None, purpose="embed"):
        self._meter(run_id, purpose, "text-embedding-3-small", 10 * len(texts), 0)
        return [fake_vector(t) for t in texts]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, azure_openai_endpoint="https://fake.example", azure_openai_api_key="x",
                    sqlite_path=tmp_path / "t.db", chroma_path=tmp_path / "chroma",
                    chunk_size=200, chunk_overlap=20, retrieval_top_k=4)


@pytest.fixture
def db(settings) -> Database:
    d = Database(settings.sqlite_path)
    yield d
    d.close()


@pytest.fixture
def llm(settings, db) -> FakeLLM:
    return FakeLLM(settings, db)


@pytest.fixture
def store(settings, llm, db) -> PolicyStore:
    return PolicyStore(settings, llm, db)


@pytest.fixture
def prompts(settings) -> PromptStore:
    return PromptStore(settings.prompts_dir)


@pytest.fixture
def agent(settings, llm, store, prompts, db) -> ClaimAgent:
    return ClaimAgent(settings, llm, store, prompts, db)


@pytest.fixture
def client(settings, monkeypatch):
    """TestClient with the real app wiring but the Azure client swapped for FakeLLM."""
    import app.deps as deps

    def fake_llm_factory(s, d):
        return FakeLLM(s, d)

    monkeypatch.setattr(deps, "LLMClient", fake_llm_factory)
    with TestClient(create_app(settings)) as c:
        yield c
