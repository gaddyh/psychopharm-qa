"""Application configuration via pydantic-settings.

Reads from environment variables and .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+psycopg://psych:psych@localhost:5432/psychopharm"
    )

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)

    # App
    streamlit_server_port: int = Field(default=8501)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="local")  # local / staging / production

    # Paths
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def evals_dir(self) -> Path:
        return PROJECT_ROOT / "evals"

    # Source file paths
    @property
    def nbn_json_path(self) -> Path:
        return self.raw_dir / "nbn_drugs.json"

    @property
    def stahl_pdf_path(self) -> Path:
        return self.raw_dir / "Prescriber's Guide_ Stahl's Essential Psychopharmacology.pdf"

    @property
    def kaplan_pdf_path(self) -> Path:
        return self.raw_dir / "Robert Boland, Marcia L. Verdiun - Kaplan.pdf"

    # Kaplan Chapter 33 page range (PDF page indices, 0-based)
    kaplan_ch33_start_page: int = Field(default=9786)
    kaplan_ch33_end_page: int = Field(default=11300)


def get_settings() -> Settings:
    """Cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


_settings: Settings | None = None
