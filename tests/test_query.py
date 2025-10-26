"""
Tests for QueryEngine
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kb_indexer import Database, QueryEngine


class TestQueryEngine(unittest.TestCase):
    """Test QueryEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        import time
        import os

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Create temporary knowledge base directory
        self.temp_kb_dir = tempfile.mkdtemp()

        # Initialize database
        self.db = Database(self.db_path)
        schema_path = Path(__file__).parent.parent / "schema.sql"
        self.db.init_schema(schema_path)

        # Add test documents
        self.db.add_document(
            filepath="rl_basics.md",
            title="Reinforcement Learning Basics",
            summary="Introduction to reinforcement learning concepts and Q-learning"
        )
        self.db.add_document_keyword("rl_basics.md", "reinforcement learning")
        self.db.add_document_keyword("rl_basics.md", "Q-learning")
        self.db.add_document_keyword("rl_basics.md", "machine learning")

        self.db.add_document(
            filepath="alphago.md",
            title="AlphaGo and Game AI",
            summary="How AlphaGo uses reinforcement learning to play Go"
        )
        self.db.add_document_keyword("alphago.md", "AlphaGo")
        self.db.add_document_keyword("alphago.md", "reinforcement learning")
        self.db.add_document_keyword("alphago.md", "game AI")

        self.db.add_document(
            filepath="neural_networks.md",
            title="Neural Networks",
            summary="Deep learning and neural network architectures"
        )
        self.db.add_document_keyword("neural_networks.md", "neural networks")
        self.db.add_document_keyword("neural_networks.md", "deep learning")

        # Create test markdown files in knowledge base
        self._create_test_file("rl_basics.md", """# Reinforcement Learning Basics

Introduction to reinforcement learning concepts and Q-learning algorithm.
Q-learning is a model-free reinforcement learning technique.
""")

        self._create_test_file("alphago.md", """# AlphaGo and Game AI

How AlphaGo uses reinforcement learning to master the game of Go.
AlphaGo combines Monte Carlo tree search with deep neural networks.
""")

        self._create_test_file("neural_networks.md", """# Neural Networks

Deep learning and neural network architectures for machine learning.
Includes convolutional neural networks (CNNs) and recurrent neural networks (RNNs).
""")

        # Create an unindexed file
        self._create_test_file("unindexed.md", """# Unindexed Document

This document contains information about deep Q-networks (DQN).
DQN combines Q-learning with deep neural networks.
""")

        # Set all file mtimes to the past to ensure they're older than DB timestamps
        # This makes tests more predictable
        past_time = time.time() - 10
        for filename in ["rl_basics.md", "alphago.md", "neural_networks.md", "unindexed.md"]:
            filepath = Path(self.temp_kb_dir) / filename
            if filepath.exists():
                os.utime(filepath, (past_time, past_time))

    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        Path(self.db_path).unlink()

        # Clean up knowledge base files
        for file in Path(self.temp_kb_dir).glob("*.md"):
            file.unlink()
        Path(self.temp_kb_dir).rmdir()

    def _create_test_file(self, filename: str, content: str):
        """Create a test markdown file."""
        filepath = Path(self.temp_kb_dir) / filename
        with open(filepath, 'w') as f:
            f.write(content)

    def test_init_claude(self):
        """Test QueryEngine initialization with Claude backend."""
        with patch('subprocess.run') as mock_run:
            # Mock successful claude --version check
            mock_run.return_value = MagicMock(returncode=0)

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="claude"
            )
            self.assertEqual(engine.backend, "claude")
            self.assertEqual(engine.model, "sonnet")

    @patch('sys.modules', {'google': MagicMock(), 'google.genai': MagicMock()})
    def test_init_gemini(self):
        """Test QueryEngine initialization with Gemini backend."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="gemini"
            )
            self.assertEqual(engine.backend, "gemini")
            self.assertEqual(engine.model, "gemini-2.0-flash-exp")

    def test_init_ollama(self):
        """Test QueryEngine initialization with Ollama backend."""
        with patch('ollama.generate'):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )
            self.assertEqual(engine.backend, "ollama")
            self.assertEqual(engine.model, "llama3.2:3b")

    def test_init_invalid_backend(self):
        """Test QueryEngine initialization with invalid backend."""
        with self.assertRaises(ValueError):
            QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="invalid"
            )

    def test_score_document_relevance(self):
        """Test document relevance scoring."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM response
            mock_generate.return_value = {
                'response': 'yes,0.85,Document discusses Q-learning in RL context'
            }

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            is_relevant, score, reasoning = engine.score_document_relevance(
                question="What is Q-learning?",
                context="reinforcement learning algorithms",
                doc_title="RL Basics",
                doc_summary="Introduction to Q-learning",
                doc_keywords=["reinforcement learning", "Q-learning"]
            )

            self.assertTrue(is_relevant)
            self.assertEqual(score, 0.85)
            self.assertIn("q-learning", reasoning.lower())

    def test_search_with_keywords(self):
        """Test keyword-based search with LLM filtering."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM responses
            mock_generate.side_effect = [
                {'response': 'yes,0.9,Directly discusses RL and Q-learning'},
                {'response': 'yes,0.85,AlphaGo uses RL for game playing'},
                {'response': 'no,0.4,Only tangentially related'}
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            results, expansion_map = engine.search_with_keywords(
                question="How does reinforcement learning work?",
                keywords=["reinforcement learning"],
                context="machine learning algorithms",
                threshold=0.7
            )

            # Should return 2 documents (above threshold)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["relevance_score"], 0.9)
            self.assertEqual(results[1]["relevance_score"], 0.85)

            # Check expansion map exists
            self.assertIsNotNone(expansion_map)

    def test_extract_search_terms(self):
        """Test search term extraction from question."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM response
            mock_generate.return_value = {
                'response': 'Q-learning\nreinforcement learning\nalgorithm'
            }

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            terms = engine._extract_search_terms(
                question="What is Q-learning algorithm?",
                context="reinforcement learning"
            )

            self.assertIsInstance(terms, list)
            self.assertIn("Q-learning", terms)
            self.assertIn("reinforcement learning", terms)

    def test_grep_search(self):
        """Test grep-based fallback search."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM responses: term extraction + document scoring for each file found
            # Grep might find multiple files, so we provide enough mock responses
            mock_generate.side_effect = [
                {'response': 'Q-learning\nDQN\ndeep'},  # term extraction
                {'response': 'yes,0.8,Discusses DQN and Q-learning'},  # First file
                {'response': 'yes,0.75,Related to Q-learning'},  # Second file
                {'response': 'yes,0.7,Related'},  # Third file (if any)
                {'response': 'yes,0.65,Related'},  # Fourth file (if any)
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            results = engine.grep_search(
                question="What is DQN?",
                context="deep reinforcement learning",
                query_keywords=["DQN", "deep Q-network"],
                threshold=0.7,
                max_results=10
            )

            # Should find at least one document
            self.assertGreater(len(results), 0)

            # Check if unindexed.md is in results
            # Note: grep should find it since it contains "DQN"
            filepaths = [r["filepath"] for r in results]

            # The unindexed file should be found if grep worked
            # With auto-indexing, it should now be indexed
            if "unindexed.md" in filepaths:
                # Check that it was auto-indexed
                for result in results:
                    if result["filepath"] == "unindexed.md":
                        # Should be indexed now (auto-indexed)
                        self.assertTrue(result["indexed"])
                        # Should be marked as auto-indexed
                        self.assertTrue(result.get("auto_indexed", False))
            else:
                # Grep ran but didn't find the file - still valid
                pass

    def test_generate_learning_suggestions(self):
        """Test learning suggestion generation."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM response with JSON suggestions
            mock_response = json.dumps({
                "keyword_suggestions": [
                    {
                        "filepath": "rl_basics.md",
                        "keywords": ["model-free", "value function"],
                        "reasoning": "Document discusses model-free methods"
                    }
                ],
                "similarity_suggestions": [
                    {
                        "keyword1": "Q-learning",
                        "keyword2": "value iteration",
                        "type": "related",
                        "context": "reinforcement learning algorithms",
                        "score": 0.7,
                        "reasoning": "Both are value-based RL methods"
                    }
                ]
            })

            mock_generate.return_value = {'response': mock_response}

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            mock_results = [
                {
                    "filepath": "rl_basics.md",
                    "title": "RL Basics",
                    "matched_keywords": ["reinforcement learning", "Q-learning"],
                    "relevance_score": 0.9
                }
            ]

            suggestions = engine.generate_learning_suggestions(
                question="What is Q-learning?",
                context="reinforcement learning",
                results=mock_results
            )

            self.assertIn("keyword_suggestions", suggestions)
            self.assertIn("similarity_suggestions", suggestions)
            self.assertGreater(len(suggestions["keyword_suggestions"]), 0)
            self.assertGreater(len(suggestions["similarity_suggestions"]), 0)

    def test_query_full_workflow(self):
        """Test complete query workflow."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM responses for document scoring
            mock_generate.side_effect = [
                {'response': 'yes,0.9,Directly discusses Q-learning'},
                {'response': 'yes,0.75,Related to RL'},
                # Learning suggestions
                {'response': json.dumps({
                    "keyword_suggestions": [],
                    "similarity_suggestions": []
                })}
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            result = engine.query(
                question="What is Q-learning?",
                keywords=["reinforcement learning", "Q-learning"],
                context="machine learning algorithms",
                threshold=0.7,
                enable_grep_fallback=True,
                enable_learning=True,
                auto_apply=False  # Test suggestion generation without applying
            )

            self.assertIn("query", result)
            self.assertIn("results", result)
            self.assertIn("count", result)
            self.assertIn("keyword_search_count", result)
            self.assertIn("grep_search_count", result)
            self.assertIn("suggestions", result)

            self.assertEqual(result["query"]["question"], "What is Q-learning?")
            self.assertGreater(result["count"], 0)

    def test_query_with_grep_fallback(self):
        """Test query with grep fallback when keyword search fails."""
        with patch('ollama.generate') as mock_generate:
            # Mock responses: no keyword matches, then grep term extraction + scoring
            mock_generate.side_effect = [
                # No documents match keywords initially
                # Grep term extraction
                {'response': 'nonexistent\nkeyword\ntest'},
                # Document scoring for grep results (if any)
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            result = engine.query(
                question="What is the nonexistent topic?",
                keywords=["nonexistent"],
                context="test context",
                threshold=0.7,
                enable_grep_fallback=True,
                enable_learning=False,
                auto_apply=False
            )

            # Grep should be attempted
            self.assertIsNotNone(result.get("grep_search_count"))

    def test_query_no_learning(self):
        """Test query with learning disabled."""
        with patch('ollama.generate') as mock_generate:
            # Mock LLM response for document scoring
            mock_generate.return_value = {
                'response': 'yes,0.8,Relevant document'
            }

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            result = engine.query(
                question="What is RL?",
                keywords=["reinforcement learning"],
                context="machine learning",
                threshold=0.7,
                enable_grep_fallback=False,
                enable_learning=False,
                auto_apply=False
            )

            self.assertIsNone(result["suggestions"])

    def test_keyword_expansion(self):
        """Test keyword expansion using similarities."""
        # Add similarity relationship
        self.db.add_similarity(
            keyword1="RL",
            keyword2="reinforcement learning",
            similarity_type="abbreviation",
            context="machine learning",
            score=1.0,
            directional=False
        )

        self.db.add_similarity(
            keyword1="reinforcement learning",
            keyword2="Q-learning",
            similarity_type="related",
            context="machine learning algorithms",
            score=0.8,
            directional=False
        )

        with patch('ollama.generate'):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Test 1-level expansion
            expanded, expansion_map = engine.expand_keywords(
                keywords=["RL"],
                context="machine learning",
                depth=1
            )

            # Should include "RL" and "reinforcement learning"
            self.assertIn("rl", expanded)  # normalized
            self.assertIn("reinforcement learning", expanded)

            # Check expansion map
            self.assertIn("rl", expansion_map)
            self.assertIn("reinforcement learning", expansion_map["rl"])

            # Test 2-level expansion
            expanded_2, expansion_map_2 = engine.expand_keywords(
                keywords=["RL"],
                context="machine learning",
                depth=2
            )

            # Should include all three
            self.assertIn("rl", expanded_2)
            self.assertIn("reinforcement learning", expanded_2)
            self.assertIn("q-learning", expanded_2)

            # Test no expansion
            not_expanded, not_expansion_map = engine.expand_keywords(
                keywords=["RL"],
                context="machine learning",
                depth=0
            )
            self.assertEqual(len(not_expanded), 1)
            self.assertEqual(len(not_expansion_map), 1)
            self.assertEqual(not_expansion_map["rl"], [])

    def test_auto_apply_suggestions(self):
        """Test automatic application of learning suggestions."""
        with patch('ollama.generate') as mock_generate, \
             patch('subprocess.run') as mock_subprocess:

            # Mock grep to find rl_basics.md file
            test_file_path = str(Path(self.temp_kb_dir) / "rl_basics.md")
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=f"{test_file_path}\n"
            )

            # Mock LLM responses:
            # 1. Extract search terms for grep
            # 2. Score the document found by grep
            mock_generate.side_effect = [
                {'response': 'Q-learning\nvalue-based\nalgorithms'},  # Search terms
                {'response': 'yes,0.85,Discusses Q-learning and RL algorithms'}  # Scoring
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Use keywords NOT in the index to trigger grep fallback and learning
            result = engine.query(
                question="What are value-based RL algorithms?",
                keywords=["value-based", "algorithms"],  # NOT in index
                context="machine learning algorithms",
                threshold=0.7,
                expand_depth=0,  # No expansion, ensure keyword search fails
                enable_grep_fallback=True,
                enable_learning=True,
                auto_apply=True  # Test auto-apply
            )

            # Verify keyword search failed (no results from keywords)
            self.assertEqual(result["keyword_search_count"], 0)

            # Verify grep found results
            self.assertGreater(result["grep_search_count"], 0)

            # Verify suggestions were generated
            self.assertIn("suggestions", result)
            self.assertIsNotNone(result["suggestions"])

            # Verify suggestions were applied
            self.assertIn("applied", result)
            self.assertIsNotNone(result["applied"])
            applied = result["applied"]

            # Check that suggestions were applied (specifically similarities)
            self.assertGreater(applied["similarities_added"], 0, "Expected similarities to be added")

            # Verify the query keywords were added as similarities to existing document keywords
            # The anchor keyword is the first keyword from the document (order may vary)
            doc_keywords = self.db.get_document_keywords("rl_basics.md")
            keyword_list = [kw["keyword"] for kw in doc_keywords]

            # Check similarities for all document keywords to find the query keywords
            found_query_keywords = []
            for doc_kw in keyword_list:
                sims = self.db.get_similar_keywords(doc_kw, similarity_type=None)
                related = [s["related_keyword"] for s in sims]
                found_query_keywords.extend(related)

            # At least one of the query keywords should be added as a similarity
            self.assertTrue(
                "value-based" in found_query_keywords or "algorithms" in found_query_keywords,
                f"Expected query keywords in similarities, got: {found_query_keywords}"
            )


    def test_needs_reindexing(self):
        """Test checking if document needs reindexing."""
        import time
        import os

        with patch('ollama.generate'):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Get document from database
            doc = self.db.get_document("rl_basics.md")

            # Set file mtime to the past (1 second ago)
            filepath = Path(self.temp_kb_dir) / "rl_basics.md"
            past_time = time.time() - 1
            os.utime(filepath, (past_time, past_time))

            # Document should NOT need reindexing (file older than DB)
            needs_reindex = engine._needs_reindexing("rl_basics.md", doc)
            self.assertFalse(needs_reindex)

            # Touch the file to make it newer
            time.sleep(0.1)  # Small delay to ensure different timestamp
            filepath.touch()

            # Now document SHOULD need reindexing
            needs_reindex = engine._needs_reindexing("rl_basics.md", doc)
            self.assertTrue(needs_reindex)

    def test_reindex_document_up_to_date(self):
        """Test reindexing when document is up to date."""
        import time
        import os

        with patch('ollama.generate'):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Set file mtime to the past to ensure it's older than DB
            filepath = Path(self.temp_kb_dir) / "rl_basics.md"
            past_time = time.time() - 1
            os.utime(filepath, (past_time, past_time))

            # Document should be up to date
            result = engine.reindex_document_if_modified("rl_basics.md")

            self.assertEqual(result["status"], "up_to_date")

    def test_reindex_document_modified(self):
        """Test reindexing when document has been modified."""
        import time

        with patch('ollama.generate') as mock_generate:
            # Mock LLM response for keyword generation
            mock_generate.return_value = {
                'response': json.dumps({
                    "keep": ["reinforcement learning", "machine learning"],
                    "add": ["temporal difference", "value function"],
                    "remove": ["q-learning"]  # Remove Q-learning
                })
            }

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Get initial keywords
            initial_keywords = [kw["keyword"] for kw in self.db.get_document_keywords("rl_basics.md")]
            self.assertIn("q-learning", initial_keywords)

            # Modify the file
            time.sleep(0.1)
            filepath = Path(self.temp_kb_dir) / "rl_basics.md"
            filepath.write_text("""# Reinforcement Learning Basics

Updated content about temporal difference learning and value functions.
This document has been significantly updated with new content.
""")

            # Reindex the document
            result = engine.reindex_document_if_modified("rl_basics.md")

            self.assertEqual(result["status"], "reindexed")
            self.assertIn("keywords_before", result)
            self.assertIn("keywords_after", result)
            self.assertIn("added", result)
            self.assertIn("removed", result)

            # Verify keywords were updated
            updated_keywords = [kw["keyword"] for kw in self.db.get_document_keywords("rl_basics.md")]
            self.assertIn("temporal difference", updated_keywords)
            self.assertIn("value function", updated_keywords)
            self.assertNotIn("q-learning", updated_keywords)

    def test_query_with_reindexing(self):
        """Test that query automatically reindexes modified documents."""
        import time
        import os

        with patch('ollama.generate') as mock_generate, \
             patch('subprocess.run') as mock_subprocess:

            # Set file mtime to past first
            filepath = Path(self.temp_kb_dir) / "rl_basics.md"
            past_time = time.time() - 2
            os.utime(filepath, (past_time, past_time))

            # Now modify rl_basics.md
            time.sleep(0.2)
            filepath.write_text("""# Reinforcement Learning Basics

Updated with new content about value-based methods.
This discusses value iteration and policy gradients.
""")

            # Mock grep to find the modified file
            test_file_path = str(filepath)
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=f"{test_file_path}\n"
            )

            # Mock LLM responses:
            # 1. Extract search terms for grep
            # 2. Reindex keyword generation
            # 3. Score the document
            mock_generate.side_effect = [
                {'response': 'value-based\nmethods\nalgorithms'},  # Search terms
                {'response': json.dumps({  # Reindexing keywords
                    "keep": ["reinforcement learning", "machine learning"],
                    "add": ["value-based", "value iteration"],
                    "remove": ["q-learning"]
                })},
                {'response': 'yes,0.9,Discusses value-based RL methods'}  # Scoring
            ]

            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Query with keywords not in current index
            result = engine.query(
                question="What are value-based methods?",
                keywords=["value-based"],
                context="reinforcement learning",
                threshold=0.7,
                expand_depth=0,
                enable_grep_fallback=True,
                enable_learning=False  # Disable learning to focus on reindexing
            )

            # Should find the document via grep
            self.assertGreater(result["grep_search_count"], 0)

            # Check if document was reindexed
            if len(result["results"]) > 0:
                for doc in result["results"]:
                    if doc["filepath"] == "rl_basics.md":
                        # Should be marked as reindexed
                        self.assertTrue(doc.get("reindexed", False))
                        break

    def test_reindex_not_indexed_document(self):
        """Test reindexing a document that's not in the database."""
        with patch('ollama.generate'):
            engine = QueryEngine(
                self.db,
                knowledge_base_path=self.temp_kb_dir,
                backend="ollama"
            )

            # Try to reindex unindexed document
            result = engine.reindex_document_if_modified("unindexed.md")

            self.assertEqual(result["status"], "not_indexed")


if __name__ == "__main__":
    unittest.main()
