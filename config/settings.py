"""
config/settings.py
──────────────────
Central configuration for the entire ArXiv GraphRAG pipeline.
All settings are read from the .env file (copy .env.example → .env).
Import this module anywhere: `from config.settings import settings`
"""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic BaseSettings automatically reads from environment variables
    and the .env file. Type annotations provide validation for free.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="LLM for answer generation")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687")
    neo4j_username: str = Field("neo4j")
    neo4j_password: str = Field(...)
    neo4j_database: str = Field("neo4j")

    # ── File Paths ────────────────────────────────────────────────────────────
    data_dir: Path = Field(Path("./data"))
    pdf_dir: Path = Field(Path("./data/pdfs"))
    extracted_dir: Path = Field(Path("./data/extracted"))
    cache_dir: Path = Field(Path("./data/cache"))

    # ── arXiv ─────────────────────────────────────────────────────────────────
    arxiv_max_results: int = Field(10, ge=1, le=100)
    arxiv_download_delay: float = Field(3.0, ge=0.5, description="Seconds between downloads")

    # ── Models ────────────────────────────────────────────────────────────────
    layout_model: str = Field("detectron2")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO")
    log_file: Path = Field(Path("./logs/pipeline.log"))

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.pdf_dir,
            self.extracted_dir,
            self.cache_dir,
            self.log_file.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton — reads .env exactly once.
    Use this everywhere instead of instantiating Settings() directly.
    """
    s = Settings()
    s.ensure_dirs()
    return s


# Convenience alias used throughout the codebase:
#   from config.settings import settings
settings = get_settings()