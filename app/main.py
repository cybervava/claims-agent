"""FastAPI entrypoint. Every HTTP request is written to the audit log."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from .config import Settings, get_settings
from .deps import build_context
from .routers import admin, claims, policies, rag

logger = logging.getLogger("claims_agent")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = build_context(settings)
        logger.info("claims-agent ready: chat=%s embed=%s db=%s rag=%s", settings.chat_deployment,
                    settings.embedding_deployment, settings.sqlite_path,
                    settings.rag_chat_model if settings.rag_enabled else "disabled")
        yield
        app.state.ctx.db.close()

    app = FastAPI(title="Insurance Claims Agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def audit_requests(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request.state.request_id
            return response
        finally:
            if request.url.path not in ("/health", "/docs", "/openapi.json"):
                request.app.state.ctx.db.audit(
                    request.headers.get("x-actor", "anonymous"), "http.request",
                    resource_type="endpoint", resource_id=f"{request.method} {request.url.path}",
                    status=str(status), request_id=request.state.request_id,
                    details={"latency_ms": int((time.perf_counter() - started) * 1000),
                             "client": request.client.host if request.client else None})

    @app.get("/health", tags=["admin"])
    def health() -> dict:
        ctx = app.state.ctx
        return {"status": "ok", "chat_model": settings.chat_deployment,
                "embedding_model": settings.embedding_deployment, "kb": ctx.store.stats(),
                "rag": {"enabled": settings.rag_enabled, "model": settings.rag_chat_model,
                        "index": settings.azure_search_index or None}}

    @app.get("/kb", include_in_schema=False)
    def kb_browser() -> FileResponse:
        return FileResponse(STATIC_DIR / "kb.html", media_type="text/html")

    @app.get("/rag", include_in_schema=False)
    def rag_ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "rag.html", media_type="text/html")

    app.include_router(policies.router)
    app.include_router(claims.router)
    app.include_router(rag.router)
    app.include_router(admin.router)
    return app


app = create_app()
