"""Claim validation agent: extract -> retrieve -> validate. Every step, LLM call,
memory write and decision is persisted to SQLite for cost and audit purposes."""
import json
import time

from .config import Settings
from .db import Database
from .llm import LLMClient
from .prompts import PromptStore
from .schemas import ClaimRecord, SearchHit, ValidateResponse, ValidationResult
from .vectorstore import PolicyStore

AGENT_NAME = "claim-validator"


class ClaimAgent:
    def __init__(self, settings: Settings, llm: LLMClient, store: PolicyStore, prompts: PromptStore, db: Database):
        self.settings = settings
        self.llm = llm
        self.store = store
        self.prompts = prompts
        self.db = db

    def run(self, claim_text: str, *, filename: str, session_id: str, actor: str,
            request_id: str | None = None) -> ValidateResponse:
        run_id = self.db.create_run(session_id, AGENT_NAME, actor, filename)
        self.db.audit(actor, "claim.validate.start", resource_type="run", resource_id=run_id,
                      request_id=request_id, details={"session_id": session_id, "filename": filename})
        try:
            claim = self._extract(run_id, claim_text)
            hits = self._retrieve(run_id, claim)
            result = self._validate(run_id, session_id, claim, hits)
            self._remember(run_id, session_id, claim, result)
        except Exception as exc:
            self.db.finish_run(run_id, "failed", error=str(exc))
            self.db.audit(actor, "claim.validate.failed", resource_type="run", resource_id=run_id,
                          request_id=request_id, status="error", details={"error": str(exc)})
            raise
        self.db.finish_run(run_id, "completed", output={"decision": result.decision,
                                                         "confidence": result.confidence})
        run = self.db.get_run(run_id) or {}
        self.db.audit(actor, "claim.validate.completed", resource_type="run", resource_id=run_id,
                      request_id=request_id, details={"decision": result.decision, "confidence": result.confidence,
                                                      "cost_usd": run.get("total_cost_usd")})
        return ValidateResponse(run_id=run_id, session_id=session_id, claim=claim, validation=result,
                                evidence=hits, cost_usd=run.get("total_cost_usd", 0.0),
                                prompt_tokens=run.get("total_prompt_tokens", 0),
                                completion_tokens=run.get("total_completion_tokens", 0))

    # ---- steps -------------------------------------------------------------
    def _extract(self, run_id: str, claim_text: str) -> ClaimRecord:
        started = time.perf_counter()
        system = self.prompts.render("claim_extractor")
        claim, _ = self.llm.chat_structured(
            [{"role": "system", "content": system.text},
             {"role": "user", "content": f"Claim file:\n\n{claim_text}"}],
            ClaimRecord, run_id=run_id, purpose="extract_claim")
        self.db.add_step(run_id, 1, "extract", prompt_name=system.name, prompt_version=system.version,
                         input_data={"chars": len(claim_text)}, output_data=claim.model_dump(),
                         latency_ms=int((time.perf_counter() - started) * 1000))
        return claim

    def _retrieve(self, run_id: str, claim: ClaimRecord) -> list[SearchHit]:
        started = time.perf_counter()
        queries = [f"{claim.claim_type} coverage conditions exclusions limits deductible",
                   claim.description, "claim notification deadline required documents"]
        queries += [item.description for item in claim.items[:3] if item.description]
        seen: dict[str, SearchHit] = {}
        for q in queries:
            for hit in self.store.search(q, run_id=run_id):
                if hit.chunk_id not in seen or hit.score > seen[hit.chunk_id].score:
                    seen[hit.chunk_id] = hit
        hits = sorted(seen.values(), key=lambda h: h.score, reverse=True)[: self.settings.retrieval_top_k]
        self.db.add_step(run_id, 2, "retrieve", input_data={"queries": queries},
                         output_data=[{"chunk_id": h.chunk_id, "score": h.score, "filename": h.filename}
                                      for h in hits],
                         latency_ms=int((time.perf_counter() - started) * 1000))
        return hits

    def _validate(self, run_id: str, session_id: str, claim: ClaimRecord, hits: list[SearchHit]) -> ValidationResult:
        started = time.perf_counter()
        memory = self.db.get_memory(session_id, self.settings.memory_window)
        memory_text = "\n".join(f"- [{m['created_at']}] {m['role']}: {m['content']}" for m in memory) or "(none)"
        clauses = "\n\n".join(f"[chunk_id: {h.chunk_id}] (source: {h.filename})\n{h.text}" for h in hits) \
            or "(no policy clauses found in the knowledge base)"
        system = self.prompts.render("claim_validator")
        user = self.prompts.render("claim_validator_user", memory=memory_text,
                                   claim_json=json.dumps(claim.model_dump(), indent=2), clauses=clauses)
        result, _ = self.llm.chat_structured(
            [{"role": "system", "content": system.text}, {"role": "user", "content": user.text}],
            ValidationResult, run_id=run_id, purpose="validate_claim")
        self.db.add_step(run_id, 3, "validate", prompt_name=system.name, prompt_version=system.version,
                         input_data={"clause_ids": [h.chunk_id for h in hits], "memory_items": len(memory),
                                     "user_prompt_version": user.version},
                         output_data=result.model_dump(), latency_ms=int((time.perf_counter() - started) * 1000))
        return result

    def _remember(self, run_id: str, session_id: str, claim: ClaimRecord, result: ValidationResult) -> None:
        self.db.add_memory(session_id, "claim",
                           f"{claim.claim_type} on {claim.incident_date or 'unknown date'}, "
                           f"policy {claim.policy_number or 'unknown'}, claimed {claim.total_claimed_amount} "
                           f"{claim.currency or ''}: {claim.description}", run_id)
        self.db.add_memory(session_id, "decision",
                           f"{result.decision} (confidence {result.confidence:.2f}): {result.summary}", run_id)
