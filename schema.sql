-- Knowledge Base Indexer Database Schema
-- SQLite schema for storing documents, keywords, and their relationships

-- Documents table
-- Stores metadata about indexed markdown documents
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,                    -- Relative path from knowledge-base/
    title TEXT,                                       -- Document title (from H1 or metadata)
    summary TEXT,                                     -- Brief description of document
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath);

-- Keywords table
-- Stores normalized keywords
CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT UNIQUE NOT NULL,                     -- Normalized (lowercase, trimmed)
    category TEXT,                                    -- Optional grouping: primary, concepts, people, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_keywords_category ON keywords(category);

-- Document-Keyword junction table
-- Many-to-many relationship between documents and keywords
CREATE TABLE IF NOT EXISTS document_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE,
    UNIQUE(document_id, keyword_id)
);

CREATE INDEX IF NOT EXISTS idx_document_keywords_doc ON document_keywords(document_id);
CREATE INDEX IF NOT EXISTS idx_document_keywords_kw ON document_keywords(keyword_id);

-- Keyword similarities table
-- Defines relationships between keywords for AI agents to interpret
CREATE TABLE IF NOT EXISTS keyword_similarities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_id_1 INTEGER NOT NULL,
    keyword_id_2 INTEGER NOT NULL,
    similarity_type TEXT NOT NULL,                    -- synonym, abbreviation, related_concept, etc.
    context TEXT NOT NULL,                            -- Human-readable explanation for AI
    score REAL DEFAULT 0.5,                           -- Relevance score (0-1)
    directional BOOLEAN DEFAULT 0,                    -- 0=bidirectional, 1=only kw1→kw2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (keyword_id_1) REFERENCES keywords(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id_2) REFERENCES keywords(id) ON DELETE CASCADE,
    UNIQUE(keyword_id_1, keyword_id_2, similarity_type),  -- Allow multiple types per keyword pair
    CHECK(keyword_id_1 < keyword_id_2),              -- Ensure consistent ordering
    CHECK(score >= 0 AND score <= 1)                  -- Validate score range
);

CREATE INDEX IF NOT EXISTS idx_similarities_kw1 ON keyword_similarities(keyword_id_1);
CREATE INDEX IF NOT EXISTS idx_similarities_kw2 ON keyword_similarities(keyword_id_2);
CREATE INDEX IF NOT EXISTS idx_similarities_type ON keyword_similarities(similarity_type);

-- Trigger to update documents.updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_documents_timestamp
AFTER UPDATE ON documents
BEGIN
    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
