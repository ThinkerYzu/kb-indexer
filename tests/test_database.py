"""
Unit tests for database module.
"""

import unittest
import tempfile
import os
from pathlib import Path

from kb_indexer.database import Database


class TestDatabase(unittest.TestCase):
    """Test Database class."""

    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Initialize schema
        self.db = Database(self.db_path)
        schema_path = Path(__file__).parent.parent / "schema.sql"
        self.db.init_schema(schema_path)

    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        os.unlink(self.db_path)

    def test_add_document(self):
        """Test adding a document."""
        doc_id = self.db.add_document(
            filepath="test.md",
            title="Test Document",
            summary="This is a test",
        )
        self.assertIsNotNone(doc_id)
        self.assertGreater(doc_id, 0)

        # Verify document was added
        doc = self.db.get_document("test.md")
        self.assertEqual(doc["filepath"], "test.md")
        self.assertEqual(doc["title"], "Test Document")
        self.assertEqual(doc["summary"], "This is a test")

    def test_get_document_not_found(self):
        """Test getting non-existent document."""
        doc = self.db.get_document("nonexistent.md")
        self.assertIsNone(doc)

    def test_update_document(self):
        """Test updating a document."""
        self.db.add_document("test.md", "Original", "Original summary")

        success = self.db.update_document("test.md", title="Updated Title")
        self.assertTrue(success)

        doc = self.db.get_document("test.md")
        self.assertEqual(doc["title"], "Updated Title")
        self.assertEqual(doc["summary"], "Original summary")

    def test_remove_document(self):
        """Test removing a document."""
        self.db.add_document("test.md", "Test", "Summary")

        success = self.db.remove_document("test.md")
        self.assertTrue(success)

        doc = self.db.get_document("test.md")
        self.assertIsNone(doc)

    def test_list_documents(self):
        """Test listing documents."""
        self.db.add_document("doc1.md", "Doc 1", "Summary 1")
        self.db.add_document("doc2.md", "Doc 2", "Summary 2")

        docs = self.db.list_documents()
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["filepath"], "doc1.md")
        self.assertEqual(docs[1]["filepath"], "doc2.md")

    def test_add_keyword(self):
        """Test adding a keyword."""
        kw_id = self.db.add_keyword("machine learning", "primary")
        self.assertIsNotNone(kw_id)

        kw = self.db.get_keyword("machine learning")
        self.assertEqual(kw["keyword"], "machine learning")
        self.assertEqual(kw["category"], "primary")

    def test_keyword_normalization(self):
        """Test keyword normalization."""
        kw_id1 = self.db.add_keyword("Machine Learning")
        kw_id2 = self.db.add_keyword("  machine learning  ")

        # Should be the same keyword
        self.assertEqual(kw_id1, kw_id2)

        kw = self.db.get_keyword("MACHINE LEARNING")
        self.assertEqual(kw["keyword"], "machine learning")

    def test_add_document_keyword(self):
        """Test associating keyword with document."""
        self.db.add_document("test.md", "Test", "Summary")
        success = self.db.add_document_keyword("test.md", "AI", "primary")
        self.assertTrue(success)

        keywords = self.db.get_document_keywords("test.md")
        self.assertEqual(len(keywords), 1)
        self.assertEqual(keywords[0]["keyword"], "ai")

    def test_get_documents_by_keyword(self):
        """Test getting documents by keyword."""
        self.db.add_document("doc1.md", "Doc 1", "Summary 1")
        self.db.add_document("doc2.md", "Doc 2", "Summary 2")

        self.db.add_document_keyword("doc1.md", "AI")
        self.db.add_document_keyword("doc2.md", "AI")

        docs = self.db.get_documents_by_keyword("ai")
        self.assertEqual(len(docs), 2)

    def test_add_similarity(self):
        """Test adding keyword similarity."""
        sim_id = self.db.add_similarity(
            keyword1="RL",
            keyword2="reinforcement learning",
            similarity_type="abbreviation",
            context="RL is abbreviation for reinforcement learning",
            score=1.0,
            directional=False,
        )
        self.assertIsNotNone(sim_id)

    def test_get_similar_keywords(self):
        """Test getting similar keywords."""
        self.db.add_similarity(
            "RL",
            "reinforcement learning",
            "abbreviation",
            "RL is abbreviation",
            1.0,
        )

        similar = self.db.get_similar_keywords("RL")
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0]["related_keyword"], "reinforcement learning")
        self.assertEqual(similar[0]["similarity_type"], "abbreviation")

        # Should work in reverse too (bidirectional)
        similar = self.db.get_similar_keywords("reinforcement learning")
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0]["related_keyword"], "rl")

    def test_filter_similar_by_type(self):
        """Test filtering similar keywords by type."""
        self.db.add_similarity("RL", "reinforcement learning", "abbreviation", "Abbrev", 1.0)
        self.db.add_similarity("RL", "machine learning", "related_concept", "Related", 0.5)

        # Filter by abbreviation
        similar = self.db.get_similar_keywords("RL", similarity_type="abbreviation")
        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0]["similarity_type"], "abbreviation")

    def test_remove_similarity(self):
        """Test removing similarity."""
        self.db.add_similarity("RL", "reinforcement learning", "abbreviation", "Context", 1.0)

        success = self.db.remove_similarity("RL", "reinforcement learning")
        self.assertTrue(success)

        similar = self.db.get_similar_keywords("RL")
        self.assertEqual(len(similar), 0)

    def test_get_stats(self):
        """Test database statistics."""
        self.db.add_document("doc1.md", "Doc 1", "Summary")
        self.db.add_keyword("AI")
        self.db.add_similarity("AI", "ML", "related_concept", "Context", 0.8)

        stats = self.db.get_stats()
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["keywords"], 2)  # AI and ML
        self.assertEqual(stats["similarities"], 1)

    def test_get_keyword_stats(self):
        """Test keyword statistics."""
        self.db.add_document("doc1.md", "Doc 1", "Summary")
        self.db.add_document_keyword("doc1.md", "AI")
        self.db.add_similarity("AI", "ML", "related_concept", "Context", 0.8)

        stats = self.db.get_keyword_stats("AI")
        self.assertEqual(stats["document_count"], 1)
        self.assertEqual(stats["related_keywords_count"], 1)

    def test_cascade_delete_document(self):
        """Test cascade deletion of document removes keywords."""
        self.db.add_document("test.md", "Test", "Summary")
        self.db.add_document_keyword("test.md", "AI")

        self.db.remove_document("test.md")

        # Document should be gone
        doc = self.db.get_document("test.md")
        self.assertIsNone(doc)

        # Keyword should still exist but not be associated
        kw = self.db.get_keyword("AI")
        self.assertIsNotNone(kw)

        # No documents should have this keyword
        docs = self.db.get_documents_by_keyword("AI")
        self.assertEqual(len(docs), 0)


if __name__ == "__main__":
    unittest.main()
