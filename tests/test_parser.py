"""
Unit tests for parser module.
"""

import unittest
import tempfile
import os
import json

from kb_indexer.parser import KeywordParser, SimilarityParser, MarkdownParser


class TestKeywordParser(unittest.TestCase):
    """Test KeywordParser class."""

    def test_parse_valid_file(self):
        """Test parsing valid keywords file."""
        data = {
            "filepath": "test.md",
            "title": "Test Document",
            "summary": "Test summary",
            "keywords": ["AI", "ML", "Deep Learning"],
            "categories": {
                "primary": ["AI", "ML"],
                "concepts": ["Deep Learning"]
            }
        }

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            parsed = KeywordParser.parse_file(temp_path)
            self.assertEqual(parsed["filepath"], "test.md")
            self.assertEqual(parsed["title"], "Test Document")
            # Keywords should be normalized to lowercase
            self.assertIn("ai", parsed["keywords"])
            self.assertIn("ml", parsed["keywords"])
            self.assertIn("deep learning", parsed["keywords"])
        finally:
            os.unlink(temp_path)

    def test_parse_missing_required_field(self):
        """Test parsing file with missing required field."""
        data = {
            "filepath": "test.md",
            # Missing 'keywords' and 'summary'
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                KeywordParser.parse_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_keywords_data(self):
        """Test validation of keywords data."""
        valid_data = {
            "filepath": "test.md",
            "summary": "Summary",
            "keywords": ["AI", "ML"]
        }
        self.assertTrue(KeywordParser.validate_keywords_data(valid_data))

        # Invalid: keywords not a list
        invalid_data = {
            "filepath": "test.md",
            "summary": "Summary",
            "keywords": "AI, ML"
        }
        with self.assertRaises(ValueError):
            KeywordParser.validate_keywords_data(invalid_data)


class TestSimilarityParser(unittest.TestCase):
    """Test SimilarityParser class."""

    def test_parse_valid_file(self):
        """Test parsing valid similarities file."""
        data = {
            "similarities": [
                {
                    "keyword1": "RL",
                    "keyword2": "Reinforcement Learning",
                    "type": "abbreviation",
                    "context": "RL is abbreviation",
                    "score": 1.0,
                    "directional": False
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            parsed = SimilarityParser.parse_file(temp_path)
            self.assertEqual(len(parsed), 1)
            # Keywords should be normalized
            self.assertEqual(parsed[0]["keyword1"], "rl")
            self.assertEqual(parsed[0]["keyword2"], "reinforcement learning")
        finally:
            os.unlink(temp_path)

    def test_parse_with_defaults(self):
        """Test parsing with default values."""
        data = {
            "similarities": [
                {
                    "keyword1": "AI",
                    "keyword2": "ML",
                    "type": "related_concept",
                    "context": "Context"
                    # Missing score and directional - should use defaults
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            parsed = SimilarityParser.parse_file(temp_path)
            self.assertEqual(parsed[0]["score"], 0.5)
            self.assertEqual(parsed[0]["directional"], False)
        finally:
            os.unlink(temp_path)

    def test_validate_similarity(self):
        """Test similarity validation."""
        valid_sim = {
            "keyword1": "AI",
            "keyword2": "ML",
            "type": "related_concept",
            "context": "Context",
            "score": 0.8,
            "directional": False
        }
        self.assertTrue(SimilarityParser.validate_similarity(valid_sim))

        # Invalid: missing required field
        invalid_sim = {
            "keyword1": "AI",
            "keyword2": "ML",
            # Missing 'type' and 'context'
        }
        with self.assertRaises(ValueError):
            SimilarityParser.validate_similarity(invalid_sim)

        # Invalid: score out of range
        invalid_sim = {
            "keyword1": "AI",
            "keyword2": "ML",
            "type": "related_concept",
            "context": "Context",
            "score": 1.5
        }
        with self.assertRaises(ValueError):
            SimilarityParser.validate_similarity(invalid_sim)


class TestMarkdownParser(unittest.TestCase):
    """Test MarkdownParser class."""

    def test_extract_title(self):
        """Test extracting title from markdown."""
        markdown = """# Test Document

This is the content."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            title = MarkdownParser.extract_title(temp_path)
            self.assertEqual(title, "Test Document")
        finally:
            os.unlink(temp_path)

    def test_extract_title_with_formatting(self):
        """Test extracting title with markdown formatting."""
        markdown = "# **Bold** and *italic* title with [link](url)"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            title = MarkdownParser.extract_title(temp_path)
            # Formatting should be stripped
            self.assertEqual(title, "Bold and italic title with link")
        finally:
            os.unlink(temp_path)

    def test_extract_title_none(self):
        """Test extracting title when no H1 exists."""
        markdown = "## H2 heading\n\nContent without H1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            title = MarkdownParser.extract_title(temp_path)
            self.assertIsNone(title)
        finally:
            os.unlink(temp_path)

    def test_extract_summary(self):
        """Test extracting summary from markdown."""
        markdown = """# Title

This is the first paragraph that should be extracted as the summary.

This is the second paragraph."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            summary = MarkdownParser.extract_summary(temp_path)
            self.assertIn("first paragraph", summary)
            self.assertNotIn("second paragraph", summary)
        finally:
            os.unlink(temp_path)

    def test_extract_summary_with_max_length(self):
        """Test summary truncation."""
        markdown = """# Title

""" + "A" * 1000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            summary = MarkdownParser.extract_summary(temp_path, max_length=100)
            self.assertLessEqual(len(summary), 100)
            self.assertTrue(summary.endswith("..."))
        finally:
            os.unlink(temp_path)

    def test_get_document_info(self):
        """Test getting document info."""
        markdown = """# Test Title

This is the summary paragraph."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown)
            temp_path = f.name

        try:
            info = MarkdownParser.get_document_info(temp_path)
            self.assertEqual(info["title"], "Test Title")
            self.assertIn("summary paragraph", info["summary"])
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
