"""Pydantic models: LLM structured-output schemas and API request/response bodies."""
from typing import Literal

from pydantic import BaseModel, Field


# ---- LLM structured outputs (strict JSON schema) ---------------------------
class ClaimItem(BaseModel):
    description: str
    amount: float | None


class ClaimRecord(BaseModel):
    policy_number: str | None
    claimant_name: str | None
    claim_type: str = Field(description="e.g. motor collision, home fire, theft, medical, travel")
    incident_date: str | None = Field(description="ISO date if determinable")
    reported_date: str | None
    location: str | None
    total_claimed_amount: float | None
    currency: str | None
    description: str = Field(description="Concise factual summary of the incident")
    items: list[ClaimItem]
    supporting_documents: list[str]
    red_flags: list[str] = Field(description="Anything inconsistent or suspicious in the claim file itself")


class Finding(BaseModel):
    rule: str = Field(description="The policy condition, exclusion or limit being checked")
    status: Literal["PASS", "FAIL", "UNCLEAR"]
    evidence_chunk_ids: list[str] = Field(description="chunk_id values from the supplied policy clauses")
    reasoning: str


class ValidationResult(BaseModel):
    decision: Literal["APPROVE", "REJECT", "NEEDS_REVIEW"]
    confidence: float = Field(description="0.0 – 1.0")
    summary: str
    findings: list[Finding]
    covered_amount_estimate: float | None
    missing_information: list[str]
    recommended_next_steps: list[str]


# ---- API bodies -------------------------------------------------------------
class IngestTextRequest(BaseModel):
    filename: str
    text: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    sha256: str
    skipped_duplicate: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    where: dict[str, str | int | float | bool] | None = None


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    score: float
    text: str
    metadata: dict


class ValidateTextRequest(BaseModel):
    claim_text: str
    session_id: str | None = None
    filename: str = "claim.txt"


class ValidateResponse(BaseModel):
    run_id: str
    session_id: str
    claim: ClaimRecord
    validation: ValidationResult
    evidence: list[SearchHit]
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
