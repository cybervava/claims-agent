"""Claim validation endpoints."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..db import new_id
from ..deps import AppContext, get_actor, get_ctx, get_request_id
from ..ingest import UnsupportedFileError, extract_text
from ..llm import LLMError
from ..schemas import ValidateResponse, ValidateTextRequest

router = APIRouter(prefix="/claims", tags=["claims"])


def _validate(ctx: AppContext, text: str, filename: str, session_id: str | None, actor: str,
              request_id: str) -> ValidateResponse:
    if not text.strip():
        raise HTTPException(400, "claim file contains no text")
    try:
        return ctx.agent.run(text, filename=filename, session_id=session_id or new_id("sess"), actor=actor,
                             request_id=request_id)
    except LLMError as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc


@router.post("/validate", response_model=ValidateResponse)
async def validate_file(file: UploadFile = File(...), session_id: str | None = Form(None),
                        ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                        request_id: str = Depends(get_request_id)) -> ValidateResponse:
    """Upload a claim file (pdf/docx/txt/md/json). Pass `session_id` to keep memory across submissions."""
    data = await file.read()
    if len(data) > ctx.settings.max_upload_bytes:
        raise HTTPException(413, f"file exceeds {ctx.settings.max_upload_bytes} bytes")
    try:
        text = extract_text(file.filename or "claim.txt", data)
    except UnsupportedFileError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _validate(ctx, text, file.filename or "claim.txt", session_id, actor, request_id)


@router.post("/validate-text", response_model=ValidateResponse)
def validate_text(body: ValidateTextRequest, ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                  request_id: str = Depends(get_request_id)) -> ValidateResponse:
    return _validate(ctx, body.claim_text, body.filename, body.session_id, actor, request_id)


@router.get("/runs")
def list_runs(limit: int = 50, session_id: str | None = None, ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    return ctx.db.list_runs(limit=min(limit, 500), session_id=session_id)


@router.get("/runs/{run_id}")
def get_run(run_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    run = ctx.db.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run
