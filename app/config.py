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


@lru_cache
def get_settings() -> Settings:
    return Settings()
