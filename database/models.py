"""
MaintiDoc - SQLAlchemy ORM Models
Defines tables for documents, text chunks, images, and tables with PGVector integration.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

from config import settings

Base = declarative_base()


class Document(Base):
    """Master document model with SHA-256 idempotency hash."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    total_pages = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    doc_metadata = Column(JSON, default=dict)

    # Relationships
    text_chunks = relationship("TextChunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("ImageMetadata", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("TableChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename='{self.filename}' pages={self.total_pages}>"


class TextChunk(Base):
    """Text chunks extracted from pages with semantic text embeddings (384-d)."""

    __tablename__ = "text_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.TEXT_EMBEDDING_DIM), nullable=True)
    chunk_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="text_chunks")

    def __repr__(self) -> str:
        return f"<TextChunk id={self.id} doc_id={self.document_id} page={self.page_number}>"


class ImageMetadata(Base):
    """Extracted figures, diagrams and photos with CLIP (512-d) and caption (384-d) vectors."""

    __tablename__ = "image_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    image_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False, index=True)
    file_path = Column(Text, nullable=False)
    caption = Column(Text, nullable=True)
    clip_embedding = Column(Vector(settings.CLIP_EMBEDDING_DIM), nullable=True)
    caption_embedding = Column(Vector(settings.TEXT_EMBEDDING_DIM), nullable=True)
    image_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="images")

    def __repr__(self) -> str:
        return f"<ImageMetadata id={self.id} doc_id={self.document_id} page={self.page_number} path='{self.file_path}'>"


class TableChunk(Base):
    """Extracted tables with markdown and structured JSON representations (384-d)."""

    __tablename__ = "table_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    table_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False, index=True)
    markdown_content = Column(Text, nullable=False)
    table_json = Column(JSON, default=dict)
    embedding = Column(Vector(settings.TEXT_EMBEDDING_DIM), nullable=True)
    table_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="tables")

    def __repr__(self) -> str:
        return f"<TableChunk id={self.id} doc_id={self.document_id} page={self.page_number}>"
