"""Application singletons wired at startup, exposed to routers as FastAPI dependencies."""
from dataclasses import dataclass

from fastapi import Request

from .agent import ClaimAgent
from .config import Settings
from .db import Database
from .llm import LLMClient
from .prompts import PromptStore
from .rag import RagChat, RagService
from .search import AzureSearchClient
from .vectorstore import PolicyStore


@dataclass
class AppContext:
    settings: Settings
    db: Database
    llm: LLMClient
    prompts: PromptStore
    store: PolicyStore
    agent: ClaimAgent
    rag: RagService | None


def build_rag(settings: Settings, prompts: PromptStore, db: Database) -> RagService | None:
    if not settings.rag_enabled:
        return None
    search = AzureSearchClient(settings.azure_search_endpoint, settings.azure_search_index,
                               settings.azure_search_api_key, api_version=settings.azure_search_api_version)
    return RagService(settings, search, RagChat(settings, db), prompts, db)


def build_context(settings: Settings) -> AppContext:
    db = Database(settings.sqlite_path)
    llm = LLMClient(settings, db)
    prompts = PromptStore(settings.prompts_dir)
    store = PolicyStore(settings, llm, db)
    agent = ClaimAgent(settings, llm, store, prompts, db)
    rag = build_rag(settings, prompts, db)
    return AppContext(settings, db, llm, prompts, store, agent, rag)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_actor(request: Request) -> str:
    return request.headers.get("x-actor", "anonymous")


def get_request_id(request: Request) -> str:
    return request.state.request_id
