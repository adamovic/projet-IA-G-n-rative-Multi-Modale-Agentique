"""
MaintiDoc - Central Configuration Module
Provides strongly typed settings, paths, and environment validation.
"""

from pathlib import Path
from typing import Literal
import os
import logging
from pydantic_settings import BaseSettings
from pydantic import Field


# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Project Information
    APP_NAME: str = "MaintiDoc"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # PostgreSQL & PGVector
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="maintdoc_db", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="postgres", env="POSTGRES_PASSWORD")

    @property
    def database_url(self) -> str:
        """Construct standard SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="qwen2.5:1.5b", env="OLLAMA_MODEL")
    OLLAMA_TEMPERATURE: float = Field(default=0.1, env="OLLAMA_TEMPERATURE")
    OLLAMA_TIMEOUT: int = Field(default=120, env="OLLAMA_TIMEOUT")

    # Embedding Models (CPU-friendly for Intel i5)
    TEXT_EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="TEXT_EMBEDDING_MODEL"
    )
    TEXT_EMBEDDING_DIM: int = Field(default=384, env="TEXT_EMBEDDING_DIM")

    CLIP_MODEL_NAME: str = Field(default="ViT-B-32", env="CLIP_MODEL_NAME")
    CLIP_PRETRAINED: str = Field(default="openai", env="CLIP_PRETRAINED")
    CLIP_EMBEDDING_DIM: int = Field(default=512, env="CLIP_EMBEDDING_DIM")

    # Hardware & Performance
    DEVICE: str = Field(default="cpu", env="DEVICE")
    CHUNK_SIZE: int = Field(default=500, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=100, env="CHUNK_OVERLAP")
    TOP_K_RETRIEVAL: int = Field(default=5, env="TOP_K_RETRIEVAL")

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = BASE_DIR / "data" / "documents"
    EXTRACTED_TEXT_DIR: Path = BASE_DIR / "data" / "extracted_text"
    EXTRACTED_IMAGES_DIR: Path = BASE_DIR / "data" / "extracted_images"
    EXTRACTED_TABLES_DIR: Path = BASE_DIR / "data" / "extracted_tables"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings singleton
settings = Settings()

# Ensure required directories exist
for path in [
    settings.DATA_DIR,
    settings.DOCUMENTS_DIR,
    settings.EXTRACTED_TEXT_DIR,
    settings.EXTRACTED_IMAGES_DIR,
    settings.EXTRACTED_TABLES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "maintdoc") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
