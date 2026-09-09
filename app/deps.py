"""Application singletons wired at startup, exposed to routers as FastAPI dependencies."""
from dataclasses import dataclass

from fastapi import Request

from .agent import ClaimAgent
from .config import Settings
from .db import Database
from .llm import LLMClient
from .prompts import PromptStore
from .vectorstore import PolicyStore


@dataclass
class AppContext:
    settings: Settings
    db: Database
    llm: LLMClient
    prompts: PromptStore
    store: PolicyStore
    agent: ClaimAgent


def build_context(settings: Settings) -> AppContext:
    db = Database(settings.sqlite_path)
    llm = LLMClient(settings, db)
    prompts = PromptStore(settings.prompts_dir)
    store = PolicyStore(settings, llm, db)
    agent = ClaimAgent(settings, llm, store, prompts, db)
    return AppContext(settings, db, llm, prompts, store, agent)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_actor(request: Request) -> str:
    return request.headers.get("x-actor", "anonymous")


def get_request_id(request: Request) -> str:
    return request.state.request_id
