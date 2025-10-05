"""
Unit tests for search module.
"""

import unittest
import tempfile
import os
from pathlib import Path

from kb_indexer.database import Database
from kb_indexer.search import SearchEngine


class TestSearchEngine(unittest.TestCase):
    """Test SearchEngine class."""

    def setUp(self):
        """Set up test database and search engine."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Initialize database
        self.db = Database(self.db_path)
        schema_path = Path(__file__).parent.parent / "schema.sql"
        self.db.init_schema(schema_path)

        # Create search engine
        self.search = SearchEngine(self.db)

        # Add test data
        self._setup_test_data()

    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        os.unlink(self.db_path)

    def _setup_test_data(self):
        """Set up test documents and keywords."""
        # Add documents
        self.db.add_document("doc1.md", "RL Paper", "About reinforcement learning")
        self.db.add_document("doc2.md", "ML Overview", "General machine learning")
        self.db.add_document("doc3.md", "Experience Learning", "Trial and error learning")

        # Add keywords
        self.db.add_document_keyword("doc1.md", "reinforcement learning")
        self.db.add_document_keyword("doc1.md", "RL")

        self.db.add_document_keyword("doc2.md", "machine learning")
        self.db.add_document_keyword("doc2.md", "ML")

        self.db.add_document_keyword("doc3.md", "experience learning")
        self.db.add_document_keyword("doc3.md", "trial and error")

        # Add similarities
        self.db.add_similarity("RL", "reinforcement learning", "abbreviation", "Abbrev", 1.0)
        self.db.add_similarity("reinforcement learning", "experience learning", "related_concept", "Related", 0.9)
        self.db.add_similarity("ML", "machine learning", "abbreviation", "Abbrev", 1.0)

    def test_search_by_keyword_single_result(self):
        """Test searching by single keyword."""
        results = self.search.search_by_keyword("reinforcement learning")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filepath"], "doc1.md")
        self.assertIn("reinforcement learning", results[0]["matched_keywords"])

    def test_search_by_keyword_multiple_results(self):
        """Test searching by keyword with multiple results."""
        # Add another doc with same keyword
        self.db.add_document("doc4.md", "RL Tutorial", "RL basics")
        self.db.add_document_keyword("doc4.md", "reinforcement learning")

        results = self.search.search_by_keyword("reinforcement learning")
        self.assertEqual(len(results), 2)

    def test_search_by_keywords_or(self):
        """Test OR search with multiple keywords."""
        results = self.search.search_by_keywords_or(["reinforcement learning", "machine learning"])
        self.assertEqual(len(results), 2)

        filepaths = [r["filepath"] for r in results]
        self.assertIn("doc1.md", filepaths)
        self.assertIn("doc2.md", filepaths)

    def test_search_by_keywords_and(self):
        """Test AND search with multiple keywords."""
        # Add doc with both keywords
        self.db.add_document("doc5.md", "Combined", "RL and ML")
        self.db.add_document_keyword("doc5.md", "reinforcement learning")
        self.db.add_document_keyword("doc5.md", "machine learning")

        results = self.search.search_by_keywords_and(["reinforcement learning", "machine learning"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["filepath"], "doc5.md")

    def test_search_by_keywords_and_no_results(self):
        """Test AND search with no matching documents."""
        results = self.search.search_by_keywords_and(["reinforcement learning", "experience learning"])
        self.assertEqual(len(results), 0)

    def test_get_similar_keywords_for_search(self):
        """Test getting similar keywords for search."""
        similar = self.search.get_similar_keywords_for_search("reinforcement learning")
        self.assertIn("rl", similar)
        self.assertIn("experience learning", similar)

    def test_get_similar_keywords_filtered_by_type(self):
        """Test getting similar keywords filtered by type."""
        similar = self.search.get_similar_keywords_for_search("reinforcement learning", "abbreviation")
        self.assertIn("rl", similar)
        self.assertNotIn("experience learning", similar)

    def test_search_with_expansion(self):
        """Test search with automatic expansion."""
        # Search for "reinforcement learning" should also find docs with "RL"
        result = self.search.search_with_expansion(
            ["reinforcement learning"],
            expand_abbreviations=True,
            expand_related=False
        )

        # Should find doc1.md (has both "reinforcement learning" and "RL")
        self.assertGreaterEqual(len(result["results"]), 1)
        self.assertIn("rl", result["expanded_keywords"])

    def test_format_search_results(self):
        """Test formatting search results."""
        results = self.search.search_by_keyword("reinforcement learning")
        formatted = self.search.format_search_results(results, ["reinforcement learning"], "exact")

        self.assertEqual(formatted["query"]["keywords"], ["reinforcement learning"])
        self.assertEqual(formatted["query"]["mode"], "exact")
        self.assertEqual(formatted["count"], 1)
        self.assertEqual(len(formatted["results"]), 1)

    def test_format_similar_keywords(self):
        """Test formatting similar keywords."""
        similarities = self.db.get_similar_keywords("reinforcement learning")
        formatted = self.search.format_similar_keywords("reinforcement learning", similarities)

        self.assertEqual(formatted["keyword"], "reinforcement learning")
        self.assertGreater(len(formatted["similar_keywords"]), 0)
        self.assertEqual(formatted["count"], len(similarities))

        # Check structure of similar keywords
        for sim in formatted["similar_keywords"]:
            self.assertIn("keyword", sim)
            self.assertIn("similarity_type", sim)
            self.assertIn("context", sim)
            self.assertIn("score", sim)
            self.assertIn("directional", sim)

    def test_format_document_details(self):
        """Test formatting document details."""
        details = self.search.format_document_details("doc1.md")

        self.assertEqual(details["filepath"], "doc1.md")
        self.assertEqual(details["title"], "RL Paper")
        self.assertIn("keywords", details)
        self.assertGreater(len(details["keywords"]), 0)

    def test_format_document_details_not_found(self):
        """Test formatting details for non-existent document."""
        details = self.search.format_document_details("nonexistent.md")
        self.assertIsNone(details)


if __name__ == "__main__":
    unittest.main()
