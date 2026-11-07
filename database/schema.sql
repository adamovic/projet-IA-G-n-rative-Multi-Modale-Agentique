-- ==============================================================================
-- MaintiDoc Database Schema - PostgreSQL with PGVector
-- Supports separate vector spaces for Text (384-d) and CLIP Vision (512-d)
-- ==============================================================================

-- 1. Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents table (Master tracking with SHA-256 deduplication)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    total_pages INT NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    doc_metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

-- 3. Text chunks table (Sentence Transformers text embeddings: 384 dimensions)
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_number INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_text_chunks_doc_id ON text_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_text_chunks_page ON text_chunks(page_number);
-- HNSW index for fast approximate nearest neighbor cosine search
CREATE INDEX IF NOT EXISTS idx_text_chunks_embedding 
    ON text_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. Image metadata and visual vectors (CLIP ViT-B-32: 512 dimensions + caption: 384 dimensions)
CREATE TABLE IF NOT EXISTS image_metadata (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    image_index INT NOT NULL,
    page_number INT NOT NULL,
    file_path TEXT NOT NULL,
    caption TEXT,
    clip_embedding vector(512),
    caption_embedding vector(384),
    image_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_image_metadata_doc_id ON image_metadata(document_id);
CREATE INDEX IF NOT EXISTS idx_image_metadata_page ON image_metadata(page_number);
-- HNSW index on CLIP visual space
CREATE INDEX IF NOT EXISTS idx_image_clip_embedding 
    ON image_metadata USING hnsw (clip_embedding vector_cosine_ops);
-- HNSW index on caption semantic space
CREATE INDEX IF NOT EXISTS idx_image_caption_embedding 
    ON image_metadata USING hnsw (caption_embedding vector_cosine_ops);

-- 5. Table chunks table (Structured data representation: 384 dimensions)
CREATE TABLE IF NOT EXISTS table_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_index INT NOT NULL,
    page_number INT NOT NULL,
    markdown_content TEXT NOT NULL,
    table_json JSONB DEFAULT '{}'::jsonb,
    embedding vector(384),
    table_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_table_chunks_doc_id ON table_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_table_chunks_page ON table_chunks(page_number);
-- HNSW index on table embeddings
CREATE INDEX IF NOT EXISTS idx_table_chunks_embedding 
    ON table_chunks USING hnsw (embedding vector_cosine_ops);
