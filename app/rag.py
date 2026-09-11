"""Simple RAG: Azure AI Search keyword retrieval -> gpt-5-mini answer with citations.

Deliberately stateless (no session memory). Each question is one run in the
same runs/steps/llm_calls/audit tables the claim agent uses, so cost and audit
reporting cover both paths.
"""
import time
from typing import Protocol

from openai import OpenAI, OpenAIError

from .config import Settings
from .db import Database, new_id
from .llm import LLMError
from .pricing import estimate_cost_usd
from .prompts import PromptStore
from .schemas import RagAskResponse, RagSource
from .search import SearchDocument

AGENT_NAME = "policy-rag"
SNIPPET_CHARS = 300
NO_CONTEXT = "(no matching policy documents were found in the index)"


class ChatBackend(Protocol):
    def complete(self, messages: list[dict], *, run_id: str, purpose: str) -> tuple[str, dict]: ...


class SearchBackend(Protocol):
    def search(self, query: str, *, top_k: int) -> list[SearchDocument]: ...


class RagChat:
    """gpt-5-mini via the OpenAI-v1-compatible Azure endpoint. Metered into the cost ledger."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.client = OpenAI(base_url=settings.rag_chat_base_url,
                             api_key=settings.rag_chat_api_key or settings.azure_openai_api_key)

    def complete(self, messages: list[dict], *, run_id: str, purpose: str) -> tuple[str, dict]:
        started = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.rag_chat_model,
                messages=messages,
                max_completion_tokens=self.settings.rag_max_completion_tokens,
                reasoning_effort=self.settings.rag_reasoning_effort,
            )
        except OpenAIError as exc:
            raise LLMError(f"chat completion failed: {exc}") from exc
        usage = completion.usage
        model = completion.model or self.settings.rag_chat_model
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        self.db.record_llm_call(run_id=run_id, purpose=purpose, model=model, prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens, cost_usd=cost, latency_ms=latency_ms)
        content = completion.choices[0].message.content if completion.choices else None
        if not content or not content.strip():
            raise LLMError("model returned an empty answer")
        return content.strip(), {"model": model, "prompt_tokens": prompt_tokens,
                                 "completion_tokens": completion_tokens, "cost_usd": cost, "latency_ms": latency_ms}


def build_context(docs: list[SearchDocument], max_chars_per_doc: int) -> str:
    """Label each document with its title so the model can cite it as [title]."""
    if not docs:
        return NO_CONTEXT
    blocks = []
    for doc in docs:
        body = doc.content[:max_chars_per_doc]
        if len(doc.content) > max_chars_per_doc:
            body += "\n[... truncated ...]"
        blocks.append(f"[{doc.title}]\n{body}")
    return "\n\n".join(blocks)


def to_source(doc: SearchDocument) -> RagSource:
    snippet = doc.highlights[0] if doc.highlights else doc.content[:SNIPPET_CHARS]
    return RagSource(id=doc.id, title=doc.title, score=round(doc.score, 4), snippet=snippet.strip())


class RagService:
    def __init__(self, settings: Settings, search: SearchBackend, chat: ChatBackend, prompts: PromptStore,
                 db: Database):
        self.settings = settings
        self.search = search
        self.chat = chat
        self.prompts = prompts
        self.db = db

    def ask(self, question: str, *, top_k: int | None = None, session_id: str | None = None,
            actor: str = "anonymous", request_id: str | None = None) -> RagAskResponse:
        started = time.perf_counter()
        session_id = session_id or new_id("rag")
        run_id = self.db.create_run(session_id, AGENT_NAME, actor, None)
        try:
            docs = self._retrieve(run_id, question, top_k or self.settings.rag_top_k)
            answer, meta = self._answer(run_id, question, docs)
        except Exception as exc:
            self.db.finish_run(run_id, "failed", error=str(exc))
            self.db.audit(actor, "rag.ask", resource_type="run", resource_id=run_id, status="error",
                          request_id=request_id, details={"question": question, "error": str(exc)})
            raise
        self.db.finish_run(run_id, "completed", output={"answer_chars": len(answer),
                                                         "sources": [d.title for d in docs]})
        self.db.audit(actor, "rag.ask", resource_type="run", resource_id=run_id, request_id=request_id,
                      details={"question": question, "sources": [d.title for d in docs],
                               "cost_usd": meta["cost_usd"]})
        return RagAskResponse(run_id=run_id, question=question, answer=answer,
                              sources=[to_source(d) for d in docs], model=meta["model"],
                              prompt_tokens=meta["prompt_tokens"], completion_tokens=meta["completion_tokens"],
                              cost_usd=meta["cost_usd"], latency_ms=int((time.perf_counter() - started) * 1000))

    def _retrieve(self, run_id: str, question: str, top_k: int) -> list[SearchDocument]:
        started = time.perf_counter()
        docs = self.search.search(question, top_k=top_k)
        self.db.add_step(run_id, 1, "retrieve", input_data={"query": question, "top_k": top_k},
                         output_data=[{"id": d.id, "title": d.title, "score": d.score, "chars": len(d.content)}
                                      for d in docs],
                         latency_ms=int((time.perf_counter() - started) * 1000))
        return docs

    def _answer(self, run_id: str, question: str, docs: list[SearchDocument]) -> tuple[str, dict]:
        started = time.perf_counter()
        system = self.prompts.render("rag_answer")
        user = self.prompts.render("rag_answer_user", question=question,
                                   context=build_context(docs, self.settings.rag_max_chars_per_doc))
        answer, meta = self.chat.complete(
            [{"role": "system", "content": system.text}, {"role": "user", "content": user.text}],
            run_id=run_id, purpose="rag_answer")
        self.db.add_step(run_id, 2, "answer", prompt_name=system.name, prompt_version=system.version,
                         input_data={"sources": [d.title for d in docs], "user_prompt_version": user.version},
                         output_data={"answer": answer, **meta},
                         latency_ms=int((time.perf_counter() - started) * 1000))
        return answer, meta
