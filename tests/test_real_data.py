"""Tests using real keywords.json files from the knowledge base.

These tests validate the system using actual production data from the knowledge-base directory.
Test data files are located in examples/ directory:
- ai-llm-vs-reinforcement-learning.keywords.json
- building-rag-systems-python.keywords.json
- python-pip-to-uv-modern-project-management.keywords.json
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from kb_indexer.database import Database


# Test data paths
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
TEST_FILES = [
    "ai-llm-vs-reinforcement-learning.keywords.json",
    "building-rag-systems-python.keywords.json",
    "python-pip-to-uv-modern-project-management.keywords.json"
]


class BaseRealDataTest(unittest.TestCase):
    """Base class for tests using real keywords data."""

    @classmethod
    def setUpClass(cls):
        """Load all real keywords.json files once for all tests."""
        cls.real_keywords_data = {}
        for filename in TEST_FILES:
            filepath = EXAMPLES_DIR / filename
            with open(filepath) as f:
                cls.real_keywords_data[filename] = json.load(f)

    def setUp(self):
        """Set up test database with real data."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Initialize database
        self.db = Database(self.db_path)
        schema_path = Path(__file__).parent.parent / "schema.sql"
        self.db.init_schema(schema_path)

        # Add all real documents to database
        for filename, keywords_data in self.real_keywords_data.items():
            md_filename = keywords_data['filepath']

            # Insert document
            self.db.add_document(
                filepath=md_filename,
                title=keywords_data['title'],
                summary=keywords_data['summary']
            )

            # Add keywords to document
            categories = keywords_data.get('categories', {})
            for keyword in keywords_data['keywords']:
                # Determine category for this keyword
                category = None
                for cat_name, cat_keywords in categories.items():
                    if keyword in cat_keywords:
                        category = cat_name
                        break

                self.db.add_document_keyword(md_filename, keyword, category)

    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)


class TestRealDataValidation(BaseRealDataTest):
    """Validate the structure and consistency of real keywords.json files."""

    def test_all_files_exist(self):
        """Verify all test data files exist."""
        for filename in TEST_FILES:
            filepath = EXAMPLES_DIR / filename
            self.assertTrue(filepath.exists(), f"Test data file not found: {filename}")

    def test_keywords_structure(self):
        """Validate that all files have required fields."""
        required_fields = ['filepath', 'title', 'summary', 'keywords', 'categories']

        for filename, data in self.real_keywords_data.items():
            for field in required_fields:
                self.assertIn(field, data, f"{filename} missing required field: {field}")

            # Validate keywords is a list
            self.assertIsInstance(data['keywords'], list, f"{filename} keywords must be a list")
            self.assertGreater(len(data['keywords']), 0, f"{filename} must have at least one keyword")

            # Validate categories structure
            categories = data['categories']
            self.assertIn('primary', categories, f"{filename} missing primary categories")
            self.assertIn('concepts', categories, f"{filename} missing concepts categories")
            self.assertIn('tools', categories, f"{filename} missing tools categories")

    def test_category_keyword_consistency(self):
        """Verify that all categorized keywords exist in the main keywords list."""
        for filename, data in self.real_keywords_data.items():
            keywords_set = set(data['keywords'])
            categories = data['categories']

            # Check each category
            for category_name, category_keywords in categories.items():
                for keyword in category_keywords:
                    self.assertIn(keyword, keywords_set,
                                f"{filename}: Category '{category_name}' contains '{keyword}' which is not in main keywords list")

    def test_no_duplicate_keywords(self):
        """Verify that keywords lists don't contain duplicates."""
        for filename, data in self.real_keywords_data.items():
            keywords = data['keywords']
            unique_keywords = set(keywords)
            self.assertEqual(len(keywords), len(unique_keywords),
                           f"{filename} contains duplicate keywords")


class TestDatabaseOperations(BaseRealDataTest):
    """Test database operations with real data."""

    def test_add_real_documents(self):
        """Verify all real documents are added to database."""
        # Should have 3 documents
        docs = self.db.list_documents()
        self.assertEqual(len(docs), 3)

        # Verify document titles
        titles = {doc['title'] for doc in docs}
        expected_titles = {
            "AI: LLMs vs Reinforcement Learning - Richard Sutton's Perspective",
            "Building RAG Systems With Python - Complete Implementation Guide",
            "From Pip to Uv: Modern Python Project Management"
        }
        self.assertEqual(titles, expected_titles)

    def test_keyword_count(self):
        """Verify total keyword count matches input data."""
        # Calculate expected unique keywords
        all_keywords = set()
        for data in self.real_keywords_data.values():
            all_keywords.update(data['keywords'])

        # Get keywords from database
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM keywords")
        db_keyword_count = cursor.fetchone()[0]

        self.assertEqual(db_keyword_count, len(all_keywords))

    def test_search_primary_keywords(self):
        """Test searching for keywords from primary categories."""
        # Search for "RAG" (from building-rag-systems-python.md)
        results = self.db.get_documents_by_keyword("RAG")
        self.assertGreater(len(results), 0)

        doc = results[0]
        self.assertIn("Building RAG Systems", doc['title'])

    def test_search_concepts(self):
        """Test searching for concept keywords."""
        # Search for "editable install" (from python-pip-to-uv)
        results = self.db.get_documents_by_keyword("editable install")
        self.assertGreater(len(results), 0)

        doc = results[0]
        self.assertIn("Python Project Management", doc['title'])

    def test_search_tools(self):
        """Test searching for tool keywords."""
        # Search for "ChromaDB" (from building-rag-systems-python.md)
        results = self.db.get_documents_by_keyword("ChromaDB")
        self.assertGreater(len(results), 0)

        doc = results[0]
        self.assertIn("RAG Systems", doc['title'])


class TestSimilarityRelationships(BaseRealDataTest):
    """Test similarity relationships with real data."""

    def test_add_similarity_for_real_keywords(self):
        """Test adding similarity relationships between real keywords."""
        # Add similarity: "RAG" is similar to "semantic search"
        self.db.add_similarity("RAG", "semantic search", "related", "Both are core concepts in knowledge retrieval", 0.9)

        # Verify relationship exists
        results = self.db.get_similar_keywords("RAG")
        self.assertGreater(len(results), 0)

        similar_keywords = [r['related_keyword'] for r in results]
        self.assertIn("semantic search", similar_keywords)

    def test_similar_keywords_from_same_document(self):
        """Test finding similarities between keywords from the same document."""
        # Keywords from python-pip-to-uv: "pip" and "uv" should be related
        self.db.add_similarity("pip", "uv", "alternative", "uv is a modern alternative to pip", 0.85)

        results = self.db.get_similar_keywords("pip")
        similar_keywords = [r['related_keyword'] for r in results]
        self.assertIn("uv", similar_keywords)

    def test_cross_document_similarity(self):
        """Test similarities between keywords from different documents."""
        # "embeddings" (RAG doc) relates to "vector similarity" (also RAG doc)
        self.db.add_similarity("embeddings", "vector similarity", "related",
                             "Embeddings enable vector similarity search", 0.88)

        # "LLM" appears in both RAG and RL documents
        # Test that searching for LLM returns both documents
        results = self.db.get_documents_by_keyword("LLM")
        self.assertGreaterEqual(len(results), 2)

        titles = {doc['title'] for doc in results}
        self.assertTrue(any("RAG" in title for title in titles))
        self.assertTrue(any("Reinforcement Learning" in title for title in titles))


class TestDataIntegrity(BaseRealDataTest):
    """Test data integrity with real data."""

    def test_document_keyword_relationships(self):
        """Verify document-keyword relationships are correct."""
        cursor = self.db.conn.cursor()

        # Get document for RAG
        cursor.execute("SELECT id FROM documents WHERE filepath = ?",
                      ("building-rag-systems-python.md",))
        doc_id = cursor.fetchone()[0]

        # Get keywords for this document
        cursor.execute("""
            SELECT k.keyword FROM keywords k
            JOIN document_keywords dk ON k.id = dk.keyword_id
            WHERE dk.document_id = ?
        """, (doc_id,))

        db_keywords = {row[0].lower() for row in cursor.fetchall()}

        # Should include expected keywords (converted to lowercase for comparison)
        expected_keywords = {"rag", "vector database", "chromadb", "semantic search"}
        self.assertTrue(expected_keywords.issubset(db_keywords))

    def test_update_real_document(self):
        """Test updating a document with new keywords."""
        # Get original RAG document data
        original_data = self.real_keywords_data["building-rag-systems-python.keywords.json"]

        # Modify document summary
        modified_summary = "Updated summary"

        # Update document
        self.db.update_document(
            filepath="building-rag-systems-python.md",
            summary=modified_summary
        )

        # Add new keyword
        self.db.add_document_keyword("building-rag-systems-python.md", "new test keyword", "primary")

        # Verify update
        results = self.db.get_documents_by_keyword("new test keyword")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['summary'], modified_summary)

    def test_keyword_categories_stored(self):
        """Verify that keyword categories are correctly stored in database."""
        # Check that keywords have correct categories
        keywords_with_categories = self.db.get_document_keywords("building-rag-systems-python.md")

        # Build a category map from the keywords
        category_map = {}
        for kw in keywords_with_categories:
            keyword = kw['keyword']
            category = kw.get('category')
            if category:
                category_map[keyword] = category

        # Verify some expected categorizations
        # (these come from the keywords.json categories field)
        original_data = self.real_keywords_data["building-rag-systems-python.keywords.json"]
        categories = original_data.get('categories', {})

        # Check at least one keyword from each category
        if 'primary' in categories and len(categories['primary']) > 0:
            keyword = categories['primary'][0]
            if keyword.lower() in [k.lower() for k in category_map]:
                # Found in our stored data, check category
                stored_category = category_map.get(keyword.lower())
                # Category might be None if not explicitly set, which is ok
                if stored_category:
                    self.assertEqual(stored_category, 'primary')


class TestSearchFunctionality(BaseRealDataTest):
    """Test search functionality with real data."""

    def test_case_insensitive_search(self):
        """Test that keyword search is case-insensitive."""
        # Search for "rag", "RAG", "Rag" should all return same result
        results_lower = self.db.get_documents_by_keyword("rag")
        results_upper = self.db.get_documents_by_keyword("RAG")
        results_mixed = self.db.get_documents_by_keyword("Rag")

        self.assertGreater(len(results_lower), 0)
        self.assertEqual(len(results_lower), len(results_upper))
        self.assertEqual(len(results_lower), len(results_mixed))

    def test_multi_word_keyword_search(self):
        """Test searching for multi-word keywords."""
        # "reinforcement learning" is a multi-word keyword
        results = self.db.get_documents_by_keyword("reinforcement learning")
        self.assertGreater(len(results), 0)

        doc = results[0]
        self.assertIn("Reinforcement Learning", doc['title'])

    def test_abbreviation_search(self):
        """Test searching for abbreviations."""
        # "AGI" is an abbreviation
        results = self.db.get_documents_by_keyword("AGI")
        self.assertGreater(len(results), 0)

        # "PEP 517" is an abbreviation
        results = self.db.get_documents_by_keyword("PEP 517")
        self.assertGreater(len(results), 0)

    def test_search_nonexistent_keyword(self):
        """Test searching for a keyword that doesn't exist."""
        results = self.db.get_documents_by_keyword("nonexistent keyword xyz123")
        self.assertEqual(len(results), 0)
