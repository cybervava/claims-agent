"""Policy knowledge-base endpoints: ingest (embed into ChromaDB), search, list, delete."""
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..deps import AppContext, get_actor, get_ctx, get_request_id
from ..ingest import UnsupportedFileError
from ..schemas import IngestResponse, IngestTextRequest, SearchHit, SearchRequest

router = APIRouter(prefix="/policies", tags=["policies"])


def _ingest(ctx: AppContext, filename: str, data: bytes, metadata: dict, actor: str, request_id: str) -> IngestResponse:
    if len(data) > ctx.settings.max_upload_bytes:
        raise HTTPException(413, f"file exceeds {ctx.settings.max_upload_bytes} bytes")
    try:
        result = ctx.store.ingest(filename, data, metadata, actor)
    except (UnsupportedFileError, ValueError) as exc:
        ctx.db.audit(actor, "policy.ingest", resource_type="document", resource_id=filename, status="error",
                     request_id=request_id, details={"error": str(exc)})
        raise HTTPException(400, str(exc)) from exc
    ctx.db.audit(actor, "policy.ingest", resource_type="document", resource_id=result.doc_id,
                 request_id=request_id, details=result.model_dump())
    return result


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), metadata: str = Form("{}"),
                      ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                      request_id: str = Depends(get_request_id)) -> IngestResponse:
    """Upload a policy document (pdf/docx/txt/md/json). `metadata` is a JSON object string, e.g. {"product":"motor"}."""
    try:
        meta = json.loads(metadata or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"metadata must be a JSON object: {exc}") from exc
    if not isinstance(meta, dict):
        raise HTTPException(400, "metadata must be a JSON object")
    data = await file.read()
    return _ingest(ctx, file.filename or "upload", data, meta, actor, request_id)


@router.post("/ingest-text", response_model=IngestResponse)
def ingest_text(body: IngestTextRequest, ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                request_id: str = Depends(get_request_id)) -> IngestResponse:
    """Ingest raw policy text (handy for scripting / tests)."""
    return _ingest(ctx, body.filename, body.text.encode("utf-8"), body.metadata, actor, request_id)


@router.post("/search", response_model=list[SearchHit])
def search(body: SearchRequest, ctx: AppContext = Depends(get_ctx)) -> list[SearchHit]:
    return ctx.store.search(body.query, top_k=body.top_k, where=body.where)


@router.get("")
def list_documents(ctx: AppContext = Depends(get_ctx)) -> dict:
    return {"stats": ctx.store.stats(), "documents": ctx.db.list_documents()}


@router.get("/{doc_id}/chunks")
def document_chunks(doc_id: str, include_embeddings: bool = False,
                    ctx: AppContext = Depends(get_ctx)) -> list[dict]:
    """Chunks of one document in index order. `include_embeddings=true` adds dims + first 8 vector values."""
    chunks = ctx.store.chunks(doc_id, include_embeddings=include_embeddings)
    if not chunks:
        raise HTTPException(404, f"document {doc_id} not found")
    return chunks


@router.delete("/{doc_id}")
def delete_document(doc_id: str, ctx: AppContext = Depends(get_ctx), actor: str = Depends(get_actor),
                    request_id: str = Depends(get_request_id)) -> dict:
    deleted = ctx.store.delete(doc_id)
    ctx.db.audit(actor, "policy.delete", resource_type="document", resource_id=doc_id,
                 status="ok" if deleted else "not_found", request_id=request_id)
    if not deleted:
        raise HTTPException(404, f"document {doc_id} not found")
    return {"deleted": doc_id}
