"""RAG endpoints: keyword-retrieve from Azure AI Search, answer with gpt-5-mini."""
from fastapi import APIRouter, Depends, HTTPException

from ..deps import AppContext, get_actor, get_ctx, get_request_id
from ..llm import LLMError
from ..rag import RagService
from ..schemas import RagAskRequest, RagAskResponse
from ..search import SearchDocument, SearchError

router = APIRouter(prefix="/rag", tags=["rag"])


def get_rag(ctx: AppContext = Depends(get_ctx)) -> RagService:
    if ctx.rag is None:
        raise HTTPException(503, "RAG is not configured: set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, "
                                 "AZURE_SEARCH_INDEX and RAG_CHAT_BASE_URL")
    return ctx.rag


@router.post("/ask", response_model=RagAskResponse)
def ask(body: RagAskRequest, rag: RagService = Depends(get_rag), actor: str = Depends(get_actor),
        request_id: str = Depends(get_request_id)) -> RagAskResponse:
    """Answer a policy question from the Azure AI Search index, citing source documents."""
    try:
        return rag.ask(body.question, top_k=body.top_k, session_id=body.session_id, actor=actor,
                       request_id=request_id)
    except SearchError as exc:
        raise HTTPException(502, f"search error: {exc}") from exc
    except LLMError as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc


@router.post("/search", response_model=list[SearchDocument])
def search(body: RagAskRequest, rag: RagService = Depends(get_rag)) -> list[SearchDocument]:
    """Raw keyword hits from Azure AI Search (no LLM call) — useful for checking retrieval."""
    try:
        return rag.search.search(body.question, top_k=body.top_k or rag.settings.rag_top_k)
    except SearchError as exc:
        raise HTTPException(502, f"search error: {exc}") from exc
