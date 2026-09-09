"""Azure OpenAI wrapper. Every call is metered into the SQLite cost ledger."""
import time
from typing import TypeVar

from openai import AzureOpenAI
from pydantic import BaseModel

from .config import Settings
from .db import Database
from .pricing import estimate_cost_usd

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    def _meter(self, *, run_id: str | None, purpose: str, model: str, prompt_tokens: int,
               completion_tokens: int, started: float) -> dict:
        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        self.db.record_llm_call(run_id=run_id, purpose=purpose, model=model, prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens, cost_usd=cost, latency_ms=latency_ms)
        return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "cost_usd": cost, "latency_ms": latency_ms}

    def chat_structured(self, messages: list[dict], schema: type[T], *, run_id: str | None,
                        purpose: str) -> tuple[T, dict]:
        """Chat completion forced into `schema` via strict JSON-schema response_format."""
        started = time.perf_counter()
        completion = self.client.chat.completions.parse(
            model=self.settings.chat_deployment,
            messages=messages,
            response_format=schema,
            temperature=self.settings.llm_temperature,
        )
        usage = completion.usage
        meta = self._meter(run_id=run_id, purpose=purpose, model=completion.model or self.settings.chat_deployment,
                           prompt_tokens=usage.prompt_tokens if usage else 0,
                           completion_tokens=usage.completion_tokens if usage else 0, started=started)
        message = completion.choices[0].message
        if message.refusal:
            raise LLMError(f"model refused: {message.refusal}")
        if message.parsed is None:
            raise LLMError("model returned no parsable structured output")
        return message.parsed, meta

    def embed(self, texts: list[str], *, run_id: str | None = None, purpose: str = "embed") -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = self.settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            started = time.perf_counter()
            resp = self.client.embeddings.create(model=self.settings.embedding_deployment, input=chunk)
            self._meter(run_id=run_id, purpose=purpose, model=resp.model or self.settings.embedding_deployment,
                        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                        completion_tokens=0, started=started)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vectors
