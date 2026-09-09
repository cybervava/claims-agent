"""Memory, cost and audit read endpoints."""
from fastapi import APIRouter, Depends

from ..deps import AppContext, get_actor, get_ctx, get_request_id

router = APIRouter(tags=["admin"])


@router.get("/memory/{session_id}")
def get_memory(session_id: str, limit: int = 50, ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    return ctx.db.get_memory(session_id, limit=min(limit, 500))


@router.delete("/memory/{session_id}")
def clear_memory(session_id: str, ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                 request_id: str = Depends(get_request_id)) -> dict:
    deleted = ctx.db.clear_memory(session_id)
    ctx.db.audit(actor, "memory.clear", resource_type="session", resource_id=session_id, request_id=request_id,
                 details={"deleted": deleted})
    return {"session_id": session_id, "deleted": deleted}


@router.get("/costs")
def costs(ctx: AppContext = Depends(get_ctx)) -> dict:
    return ctx.db.cost_summary()


@router.get("/audit")
def audit(limit: int = 100, action: str | None = None, ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    return ctx.db.list_audit(limit=min(limit, 1000), action=action)


@router.get("/prompts")
def prompts(ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    return ctx.prompts.list()
