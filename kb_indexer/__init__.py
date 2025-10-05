"""
Knowledge Base Indexer

A pure data tool for indexing documents with keywords and semantic relationships.
Designed to provide structured data for AI agents.
"""

__version__ = "0.1.0"
__author__ = "Claude Code"

from .database import Database
from .parser import KeywordParser, MarkdownParser
from .search import SearchEngine

__all__ = ["Database", "KeywordParser", "MarkdownParser", "SearchEngine"]
