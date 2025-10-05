"""
Parser module for Knowledge Base Indexer.

Handles parsing of JSON keyword files and markdown documents.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union


class KeywordParser:
    """Parser for .keywords.json files."""

    @staticmethod
    def parse_file(filepath: Union[str, Path]) -> Dict:
        """Parse a keywords.json file.

        Args:
            filepath: Path to .keywords.json file

        Returns:
            Dictionary with parsed keyword data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If required fields are missing
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Keyword file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        if "filepath" not in data:
            raise ValueError("Missing required field: filepath")

        if "keywords" not in data:
            raise ValueError("Missing required field: keywords")

        if "summary" not in data:
            raise ValueError("Missing required field: summary")

        # Normalize keywords to lowercase
        data["keywords"] = [kw.strip().lower() for kw in data["keywords"]]

        # Normalize categories if present
        if "categories" in data:
            normalized_categories = {}
            for category, keywords in data["categories"].items():
                normalized_categories[category] = [kw.strip().lower() for kw in keywords]
            data["categories"] = normalized_categories

        return data

    @staticmethod
    def validate_keywords_data(data: Dict) -> bool:
        """Validate keywords data structure.

        Args:
            data: Keyword data dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If data is invalid
        """
        # Required fields
        required = ["filepath", "keywords", "summary"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Keywords must be a list
        if not isinstance(data["keywords"], list):
            raise ValueError("keywords must be a list")

        # All keywords must be strings
        if not all(isinstance(kw, str) for kw in data["keywords"]):
            raise ValueError("All keywords must be strings")

        # Categories must be a dict if present
        if "categories" in data and not isinstance(data["categories"], dict):
            raise ValueError("categories must be a dictionary")

        return True


class SimilarityParser:
    """Parser for similarities.json file."""

    @staticmethod
    def parse_file(filepath: Union[str, Path]) -> List[Dict]:
        """Parse a similarities.json file.

        Args:
            filepath: Path to similarities.json file

        Returns:
            List of similarity dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            ValueError: If required fields are missing
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Similarity file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "similarities" not in data:
            raise ValueError("Missing required field: similarities")

        similarities = data["similarities"]

        if not isinstance(similarities, list):
            raise ValueError("similarities must be a list")

        # Validate and normalize each similarity
        normalized = []
        for i, sim in enumerate(similarities):
            try:
                SimilarityParser.validate_similarity(sim)
                # Normalize keywords
                sim["keyword1"] = sim["keyword1"].strip().lower()
                sim["keyword2"] = sim["keyword2"].strip().lower()
                # Set defaults
                if "score" not in sim:
                    sim["score"] = 0.5
                if "directional" not in sim:
                    sim["directional"] = False
                normalized.append(sim)
            except ValueError as e:
                raise ValueError(f"Invalid similarity at index {i}: {e}")

        return normalized

    @staticmethod
    def validate_similarity(sim: Dict) -> bool:
        """Validate a single similarity entry.

        Args:
            sim: Similarity dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If similarity is invalid
        """
        # Required fields
        required = ["keyword1", "keyword2", "type", "context"]
        for field in required:
            if field not in sim:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(sim["keyword1"], str):
            raise ValueError("keyword1 must be a string")
        if not isinstance(sim["keyword2"], str):
            raise ValueError("keyword2 must be a string")
        if not isinstance(sim["type"], str):
            raise ValueError("type must be a string")
        if not isinstance(sim["context"], str):
            raise ValueError("context must be a string")

        # Validate optional fields
        if "score" in sim:
            if not isinstance(sim["score"], (int, float)):
                raise ValueError("score must be a number")
            if not (0 <= sim["score"] <= 1):
                raise ValueError("score must be between 0 and 1")

        if "directional" in sim:
            if not isinstance(sim["directional"], bool):
                raise ValueError("directional must be a boolean")

        return True


class MarkdownParser:
    """Parser for markdown documents."""

    @staticmethod
    def extract_title(filepath: Union[str, Path]) -> Optional[str]:
        """Extract title from markdown document.

        Looks for the first H1 heading (# Title).

        Args:
            filepath: Path to markdown file

        Returns:
            Extracted title or None if not found
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Markdown file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Look for H1 heading
                if line.startswith("# "):
                    title = line[2:].strip()
                    # Remove any markdown formatting from title
                    title = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", title)  # Remove links
                    title = re.sub(r"[*_`~]", "", title)  # Remove emphasis
                    return title

        return None

    @staticmethod
    def extract_summary(filepath: Union[str, Path], max_length: int = 500) -> Optional[str]:
        """Extract a summary from markdown document.

        Reads the first paragraph after the title.

        Args:
            filepath: Path to markdown file
            max_length: Maximum summary length in characters

        Returns:
            Extracted summary or None if not found
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Markdown file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Skip title (first H1)
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("# "):
                start_idx = i + 1
                break

        # Find first non-empty paragraph
        summary_lines = []
        in_paragraph = False

        for line in lines[start_idx:]:
            line = line.strip()

            # Skip empty lines before paragraph starts
            if not line and not in_paragraph:
                continue

            # Stop at headings or code blocks
            if line.startswith("#") or line.startswith("```"):
                break

            # Empty line ends paragraph
            if not line and in_paragraph:
                break

            # Accumulate paragraph lines
            if line:
                in_paragraph = True
                summary_lines.append(line)

        if not summary_lines:
            return None

        summary = " ".join(summary_lines)

        # Remove markdown formatting
        summary = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", summary)  # Links
        summary = re.sub(r"[*_`~]", "", summary)  # Emphasis

        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    @staticmethod
    def get_document_info(filepath: Union[str, Path]) -> Dict:
        """Extract title and summary from markdown document.

        Args:
            filepath: Path to markdown file

        Returns:
            Dictionary with 'title' and 'summary' keys
        """
        return {
            "title": MarkdownParser.extract_title(filepath),
            "summary": MarkdownParser.extract_summary(filepath),
        }
