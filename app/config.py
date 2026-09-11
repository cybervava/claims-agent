"""Runtime settings. Everything secret comes from .env / environment, never from code."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-10-21"
    chat_deployment: str = "gpt-4o-mini"
    embedding_deployment: str = "text-embedding-3-small"

    sqlite_path: Path = BASE_DIR / "data" / "claims_agent.db"
    chroma_path: Path = BASE_DIR / "data" / "chroma"
    chroma_collection: str = "policies"
    prompts_dir: Path = BASE_DIR / "prompts"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    embed_batch_size: int = 64
    retrieval_top_k: int = 8
    memory_window: int = 6
    llm_temperature: float = 0.0
    max_upload_bytes: int = 10 * 1024 * 1024

    # RAG over Azure AI Search + gpt-5-mini. Optional: /rag endpoints return 503 until all three
    # AZURE_SEARCH_* values and RAG_CHAT_BASE_URL are set.
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = ""
    azure_search_api_version: str = "2024-07-01"
    rag_top_k: int = 4
    rag_max_chars_per_doc: int = 8000
    rag_chat_base_url: str = ""  # e.g. https://<resource>.openai.azure.com/openai/v1
    rag_chat_api_key: str = ""  # falls back to azure_openai_api_key
    rag_chat_model: str = "gpt-5-mini"
    rag_reasoning_effort: str = "low"
    rag_max_completion_tokens: int = 4096

    @property
    def rag_enabled(self) -> bool:
        return bool(self.azure_search_endpoint and self.azure_search_api_key
                    and self.azure_search_index and self.rag_chat_base_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
