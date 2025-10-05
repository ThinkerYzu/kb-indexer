"""
Database module for Knowledge Base Indexer.

Handles all SQLite database operations including initialization, CRUD operations
for documents, keywords, and similarity relationships.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class Database:
    """SQLite database interface for knowledge base indexer."""

    def __init__(self, db_path: Union[str, Path] = "kb_index.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection with foreign key support."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def init_schema(self, schema_path: Union[str, Path]) -> None:
        """Initialize database schema from SQL file.

        Args:
            schema_path: Path to schema.sql file
        """
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        self.conn.executescript(schema_sql)
        self.conn.commit()

    # ==================== Document Operations ====================

    def add_document(
        self,
        filepath: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> int:
        """Add a new document to the database.

        Args:
            filepath: Relative path from knowledge-base/
            title: Document title
            summary: Document summary

        Returns:
            Document ID

        Raises:
            sqlite3.IntegrityError: If filepath already exists
        """
        cursor = self.conn.execute(
            """
            INSERT INTO documents (filepath, title, summary)
            VALUES (?, ?, ?)
            """,
            (filepath, title, summary),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_document(self, filepath: str) -> Optional[Dict]:
        """Get document by filepath.

        Args:
            filepath: Relative path from knowledge-base/

        Returns:
            Document data as dict or None if not found
        """
        cursor = self.conn.execute(
            """
            SELECT id, filepath, title, summary, created_at, updated_at
            FROM documents
            WHERE filepath = ?
            """,
            (filepath,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_document_by_id(self, document_id: int) -> Optional[Dict]:
        """Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document data as dict or None if not found
        """
        cursor = self.conn.execute(
            """
            SELECT id, filepath, title, summary, created_at, updated_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_document(
        self,
        filepath: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> bool:
        """Update existing document.

        Args:
            filepath: Relative path from knowledge-base/
            title: New title (None = no change)
            summary: New summary (None = no change)

        Returns:
            True if document was updated, False if not found
        """
        # Build dynamic UPDATE query
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(filepath)

        query = f"""
            UPDATE documents
            SET {', '.join(updates)}
            WHERE filepath = ?
        """

        cursor = self.conn.execute(query, params)
        self.conn.commit()
        return cursor.rowcount > 0

    def remove_document(self, filepath: str) -> bool:
        """Remove document from database.

        Args:
            filepath: Relative path from knowledge-base/

        Returns:
            True if document was removed, False if not found
        """
        cursor = self.conn.execute(
            "DELETE FROM documents WHERE filepath = ?", (filepath,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def list_documents(self) -> List[Dict]:
        """List all documents in database.

        Returns:
            List of document dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT id, filepath, title, summary, created_at, updated_at
            FROM documents
            ORDER BY filepath
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Keyword Operations ====================

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword to lowercase and trimmed.

        Args:
            keyword: Raw keyword

        Returns:
            Normalized keyword
        """
        return keyword.strip().lower()

    def add_keyword(self, keyword: str, category: Optional[str] = None) -> int:
        """Add a keyword to the database.

        Args:
            keyword: Keyword text (will be normalized)
            category: Optional category grouping

        Returns:
            Keyword ID (existing or newly created)
        """
        normalized = self._normalize_keyword(keyword)

        # Try to insert, ignore if already exists
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO keywords (keyword, category)
            VALUES (?, ?)
            """,
            (normalized, category),
        )

        # If inserted (rowcount > 0), return new ID
        if cursor.rowcount > 0:
            self.conn.commit()
            return cursor.lastrowid

        # Otherwise, fetch existing ID
        cursor = self.conn.execute(
            "SELECT id FROM keywords WHERE keyword = ?", (normalized,)
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def get_keyword(self, keyword: str) -> Optional[Dict]:
        """Get keyword by text.

        Args:
            keyword: Keyword text

        Returns:
            Keyword data as dict or None if not found
        """
        normalized = self._normalize_keyword(keyword)
        cursor = self.conn.execute(
            """
            SELECT id, keyword, category, created_at
            FROM keywords
            WHERE keyword = ?
            """,
            (normalized,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_keyword_by_id(self, keyword_id: int) -> Optional[Dict]:
        """Get keyword by ID.

        Args:
            keyword_id: Keyword ID

        Returns:
            Keyword data as dict or None if not found
        """
        cursor = self.conn.execute(
            """
            SELECT id, keyword, category, created_at
            FROM keywords
            WHERE id = ?
            """,
            (keyword_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_keywords(self) -> List[Dict]:
        """List all keywords in database.

        Returns:
            List of keyword dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT id, keyword, category, created_at
            FROM keywords
            ORDER BY keyword
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Document-Keyword Relationship Operations ====================

    def add_document_keyword(self, filepath: str, keyword: str, category: Optional[str] = None) -> bool:
        """Associate a keyword with a document.

        Args:
            filepath: Document filepath
            keyword: Keyword text
            category: Optional keyword category

        Returns:
            True if association was created, False if already exists
        """
        # Get or create document
        doc = self.get_document(filepath)
        if not doc:
            raise ValueError(f"Document not found: {filepath}")

        # Get or create keyword
        keyword_id = self.add_keyword(keyword, category)

        # Create association
        try:
            self.conn.execute(
                """
                INSERT INTO document_keywords (document_id, keyword_id)
                VALUES (?, ?)
                """,
                (doc["id"], keyword_id),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Association already exists
            return False

    def remove_document_keyword(self, filepath: str, keyword: str) -> bool:
        """Remove keyword association from document.

        Args:
            filepath: Document filepath
            keyword: Keyword text

        Returns:
            True if association was removed, False if not found
        """
        normalized = self._normalize_keyword(keyword)

        cursor = self.conn.execute(
            """
            DELETE FROM document_keywords
            WHERE document_id = (SELECT id FROM documents WHERE filepath = ?)
              AND keyword_id = (SELECT id FROM keywords WHERE keyword = ?)
            """,
            (filepath, normalized),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_document_keywords(self, filepath: str) -> List[Dict]:
        """Get all keywords for a document.

        Args:
            filepath: Document filepath

        Returns:
            List of keyword dictionaries with category
        """
        cursor = self.conn.execute(
            """
            SELECT k.id, k.keyword, k.category
            FROM keywords k
            JOIN document_keywords dk ON k.id = dk.keyword_id
            JOIN documents d ON d.id = dk.document_id
            WHERE d.filepath = ?
            ORDER BY k.keyword
            """,
            (filepath,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_documents_by_keyword(self, keyword: str) -> List[Dict]:
        """Get all documents containing a keyword.

        Args:
            keyword: Keyword text

        Returns:
            List of document dictionaries
        """
        normalized = self._normalize_keyword(keyword)

        cursor = self.conn.execute(
            """
            SELECT d.id, d.filepath, d.title, d.summary, d.created_at, d.updated_at
            FROM documents d
            JOIN document_keywords dk ON d.id = dk.document_id
            JOIN keywords k ON k.id = dk.keyword_id
            WHERE k.keyword = ?
            ORDER BY d.filepath
            """,
            (normalized,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Similarity Operations ====================

    def add_similarity(
        self,
        keyword1: str,
        keyword2: str,
        similarity_type: str,
        context: str,
        score: float = 0.5,
        directional: bool = False,
    ) -> int:
        """Add or update a keyword similarity relationship.

        Args:
            keyword1: First keyword
            keyword2: Second keyword
            similarity_type: Type of relationship
            context: Human-readable explanation
            score: Relevance score (0-1)
            directional: If True, relationship is keyword1→keyword2 only

        Returns:
            Similarity ID
        """
        # Normalize keywords
        kw1_norm = self._normalize_keyword(keyword1)
        kw2_norm = self._normalize_keyword(keyword2)

        # Get or create keywords
        kw1_id = self.add_keyword(kw1_norm)
        kw2_id = self.add_keyword(kw2_norm)

        # Ensure kw1_id < kw2_id for consistent ordering
        if kw1_id >= kw2_id:
            if kw1_id == kw2_id:
                raise ValueError("Cannot create similarity between a keyword and itself")
            kw1_id, kw2_id = kw2_id, kw1_id

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Check if similarity already exists
        existing = self.conn.execute(
            """
            SELECT id FROM keyword_similarities
            WHERE keyword_id_1 = ? AND keyword_id_2 = ?
            """,
            (kw1_id, kw2_id),
        ).fetchone()

        if existing:
            # Update existing similarity
            self.conn.execute(
                """
                UPDATE keyword_similarities
                SET similarity_type = ?, context = ?, score = ?, directional = ?
                WHERE id = ?
                """,
                (similarity_type, context, score, directional, existing["id"]),
            )
            self.conn.commit()
            return existing["id"]
        else:
            # Insert new similarity
            cursor = self.conn.execute(
                """
                INSERT INTO keyword_similarities
                    (keyword_id_1, keyword_id_2, similarity_type, context, score, directional)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kw1_id, kw2_id, similarity_type, context, score, directional),
            )
            self.conn.commit()
            return cursor.lastrowid

    def remove_similarity(self, keyword1: str, keyword2: str) -> bool:
        """Remove keyword similarity relationship.

        Args:
            keyword1: First keyword
            keyword2: Second keyword

        Returns:
            True if relationship was removed, False if not found
        """
        kw1_norm = self._normalize_keyword(keyword1)
        kw2_norm = self._normalize_keyword(keyword2)

        # Get keyword IDs
        kw1 = self.get_keyword(kw1_norm)
        kw2 = self.get_keyword(kw2_norm)

        if not kw1 or not kw2:
            return False

        kw1_id, kw2_id = kw1["id"], kw2["id"]

        # Ensure consistent ordering
        if kw1_id > kw2_id:
            kw1_id, kw2_id = kw2_id, kw1_id

        cursor = self.conn.execute(
            """
            DELETE FROM keyword_similarities
            WHERE keyword_id_1 = ? AND keyword_id_2 = ?
            """,
            (kw1_id, kw2_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_similar_keywords(
        self,
        keyword: str,
        similarity_type: Optional[str] = None,
    ) -> List[Dict]:
        """Get all keywords similar to the given keyword.

        Args:
            keyword: Keyword to find similarities for
            similarity_type: Optional filter by similarity type

        Returns:
            List of similarity dictionaries with related keyword info
        """
        kw_norm = self._normalize_keyword(keyword)
        kw = self.get_keyword(kw_norm)

        if not kw:
            return []

        kw_id = kw["id"]

        # Build query based on filtering
        if similarity_type:
            query = """
                SELECT
                    ks.id,
                    ks.similarity_type,
                    ks.context,
                    ks.score,
                    ks.directional,
                    CASE
                        WHEN ks.keyword_id_1 = ? THEN k2.keyword
                        ELSE k1.keyword
                    END as related_keyword
                FROM keyword_similarities ks
                JOIN keywords k1 ON k1.id = ks.keyword_id_1
                JOIN keywords k2 ON k2.id = ks.keyword_id_2
                WHERE (ks.keyword_id_1 = ? OR ks.keyword_id_2 = ?)
                  AND ks.similarity_type = ?
                ORDER BY ks.score DESC, related_keyword
            """
            params = (kw_id, kw_id, kw_id, similarity_type)
        else:
            query = """
                SELECT
                    ks.id,
                    ks.similarity_type,
                    ks.context,
                    ks.score,
                    ks.directional,
                    CASE
                        WHEN ks.keyword_id_1 = ? THEN k2.keyword
                        ELSE k1.keyword
                    END as related_keyword
                FROM keyword_similarities ks
                JOIN keywords k1 ON k1.id = ks.keyword_id_1
                JOIN keywords k2 ON k2.id = ks.keyword_id_2
                WHERE ks.keyword_id_1 = ? OR ks.keyword_id_2 = ?
                ORDER BY ks.score DESC, related_keyword
            """
            params = (kw_id, kw_id, kw_id)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Statistics Operations ====================

    def get_stats(self) -> Dict:
        """Get database statistics.

        Returns:
            Dictionary with counts of documents, keywords, and similarities
        """
        doc_count = self.conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()["count"]
        kw_count = self.conn.execute("SELECT COUNT(*) as count FROM keywords").fetchone()["count"]
        sim_count = self.conn.execute("SELECT COUNT(*) as count FROM keyword_similarities").fetchone()["count"]

        return {
            "documents": doc_count,
            "keywords": kw_count,
            "similarities": sim_count,
        }

    def get_keyword_stats(self, keyword: str) -> Optional[Dict]:
        """Get statistics for a specific keyword.

        Args:
            keyword: Keyword text

        Returns:
            Dictionary with document count and related keywords count
        """
        kw = self.get_keyword(keyword)
        if not kw:
            return None

        kw_id = kw["id"]

        # Count documents
        doc_count = self.conn.execute(
            """
            SELECT COUNT(*) as count
            FROM document_keywords
            WHERE keyword_id = ?
            """,
            (kw_id,),
        ).fetchone()["count"]

        # Count related keywords
        related_count = self.conn.execute(
            """
            SELECT COUNT(*) as count
            FROM keyword_similarities
            WHERE keyword_id_1 = ? OR keyword_id_2 = ?
            """,
            (kw_id, kw_id),
        ).fetchone()["count"]

        return {
            "keyword": keyword,
            "document_count": doc_count,
            "related_keywords_count": related_count,
        }
