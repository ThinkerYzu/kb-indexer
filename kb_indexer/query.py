#!/usr/bin/env python3
"""
Query Engine - Intelligent document querying with question-based filtering.

This module provides advanced query capabilities that combine keyword search,
LLM-based relevance filtering, grep fallback, and automatic learning.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

from .database import Database
from .search import SearchEngine


class QueryEngine:
    """
    Advanced query engine with LLM-based filtering and learning.
    """

    def __init__(
        self,
        db: Database,
        knowledge_base_path: str = "./knowledge-base",
        backend: Literal["claude", "gemini", "ollama"] = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize QueryEngine.

        Args:
            db: Database instance
            knowledge_base_path: Path to knowledge base directory
            backend: LLM backend to use ("ollama", "claude", or "gemini"). Default: "ollama"
            api_key: API key for Gemini. If None, reads from GEMINI_API_KEY env variable.
            model: Model name. Defaults: llama3.2:3b for Ollama, 'sonnet' for Claude, gemini-2.0-flash-exp for Gemini.
        """
        self.db = db
        self.search_engine = SearchEngine(db)
        self.knowledge_base_path = Path(knowledge_base_path)
        self.backend = backend

        # Initialize LLM backend
        if backend == "claude":
            # Claude Code CLI - check if available
            try:
                result = subprocess.run(
                    ['claude', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    raise ValueError("Claude Code CLI not found or not working")
            except (subprocess.SubprocessError, FileNotFoundError):
                raise ValueError(
                    "Claude Code CLI is required for Claude backend. "
                    "Install from: https://docs.claude.com/en/docs/claude-code"
                )

            self.model = model or 'sonnet'  # Use alias for latest sonnet model

        elif backend == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai is required for Gemini backend. "
                    "Install with: pip install google-genai"
                )

            if api_key is None:
                api_key = os.getenv('GEMINI_API_KEY')

            if not api_key:
                raise ValueError(
                    "No Gemini API key provided. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self.client = genai.Client(api_key=api_key)
            self.model = model or 'gemini-2.0-flash-exp'

        elif backend == "ollama":
            try:
                import ollama
                self.ollama = ollama
            except ImportError:
                raise ImportError(
                    "ollama is required for Ollama backend. "
                    "Install with: pip install ollama"
                )

            self.model = model or 'llama3.2:3b'

        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'claude', 'gemini', or 'ollama'.")

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with prompt and return response.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        if self.backend == "claude":
            # Use Claude Code CLI
            try:
                result = subprocess.run(
                    ['claude', '--print', '--model', self.model, '--output-format', 'text', prompt],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    raise Exception(f"Claude CLI error: {result.stderr}")
                return result.stdout.strip()
            except subprocess.TimeoutExpired:
                raise Exception("Claude CLI timeout")
            except Exception as e:
                raise Exception(f"Claude CLI failed: {e}")

        elif self.backend == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text.strip()

        elif self.backend == "ollama":
            response = self.ollama.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response'].strip()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def score_document_relevance(
        self,
        question: str,
        context: str,
        doc_title: str,
        doc_summary: str,
        doc_keywords: List[str]
    ) -> Tuple[bool, float, str]:
        """
        Score a document's relevance to the question and context using LLM.

        Args:
            question: User's question
            context: User's context/domain
            doc_title: Document title
            doc_summary: Document summary
            doc_keywords: List of document keywords

        Returns:
            Tuple of (is_relevant: bool, score: float, reasoning: str)
            - is_relevant: True if document is relevant
            - score: 0.0-1.0 relevance score
            - reasoning: Brief explanation of the score
        """
        prompt = f"""Evaluate if this document is relevant to the user's question and context.

User's Question: {question}
User's Context: {context}

Document Information:
- Title: {doc_title or "N/A"}
- Summary: {doc_summary or "N/A"}
- Keywords: {", ".join(doc_keywords)}

Task: Determine if this document would help answer the user's question within their context.

Scoring Guidelines:
- 0.9-1.0: Directly answers the question with highly relevant content
- 0.7-0.89: Contains relevant information that partially answers the question
- 0.5-0.69: Related to the topic but doesn't directly answer the question
- 0.3-0.49: Tangentially related, might provide background context
- 0.0-0.29: Not relevant to the question or context

Answer in format: yes|no,score,reasoning
- yes/no: Whether document should be included
- score: Relevance score (0.0-1.0)
- reasoning: One sentence explaining the score

Your answer:"""

        try:
            answer = self._call_llm(prompt).lower()

            # Parse response: yes|no,score,reasoning
            match = re.match(r'(yes|no),(\d+\.?\d*),(.+)', answer)

            if not match:
                # Fallback parsing
                if 'yes' in answer:
                    is_relevant = True
                    score_match = re.search(r'(\d+\.?\d*)', answer)
                    score = float(score_match.group(1)) if score_match else 0.5
                    reasoning = answer.split(',', 2)[2] if ',' in answer else "Relevant document"
                elif 'no' in answer:
                    is_relevant = False
                    score_match = re.search(r'(\d+\.?\d*)', answer)
                    score = float(score_match.group(1)) if score_match else 0.3
                    reasoning = answer.split(',', 2)[2] if ',' in answer else "Not relevant"
                else:
                    return (False, 0.3, "Unable to parse LLM response")
            else:
                is_relevant = match.group(1) == 'yes'
                score = float(match.group(2))
                reasoning = match.group(3).strip()

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return (is_relevant, score, reasoning)

        except Exception as e:
            print(f"Warning: Document scoring failed: {e}")
            return (False, 0.0, f"Error: {str(e)}")

    def expand_keywords(
        self,
        keywords: List[str],
        context: Optional[str] = None,
        depth: int = 1,
        context_threshold: float = 0.7
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Expand keywords using similarity relationships.

        Args:
            keywords: Initial list of keywords
            context: Optional context for filtering similar keywords
            depth: Number of expansion levels (0 = no expansion, 1 = one level, etc.)
            context_threshold: Minimum context match score when filtering by context

        Returns:
            Tuple of (expanded_keywords, expansion_map)
            - expanded_keywords: List of all keywords (includes original keywords)
            - expansion_map: Dict mapping each original keyword to its expanded keywords
        """
        if depth == 0:
            # No expansion - map each keyword to itself
            normalized = [self.db._normalize_keyword(kw) for kw in keywords]
            expansion_map = {self.db._normalize_keyword(kw): [] for kw in keywords}
            return normalized, expansion_map

        # Track expansion mapping: original_keyword -> [expanded keywords]
        expansion_map: Dict[str, List[str]] = {}

        # Normalize original keywords
        original_normalized = [self.db._normalize_keyword(kw) for kw in keywords]

        # Initialize expansion map
        for orig_kw in original_normalized:
            expansion_map[orig_kw] = []

        expanded = set(original_normalized)
        current_level = set(original_normalized)

        # Track which original keyword led to each expansion
        keyword_origin: Dict[str, str] = {}
        for orig_kw in original_normalized:
            keyword_origin[orig_kw] = orig_kw

        for level in range(depth):
            next_level = set()

            for keyword in current_level:
                # Get the original keyword this came from
                origin = keyword_origin.get(keyword, keyword)

                # Get all similar keywords (all types)
                similar_keywords = self.db.get_similar_keywords(keyword, similarity_type=None)

                for sim in similar_keywords:
                    related_kw = sim["related_keyword"]

                    # If context filtering is enabled, check context match
                    if context and sim.get("context"):
                        # Use simple context matching (check if context appears in similarity context)
                        # For more sophisticated matching, could use LLM but that's expensive
                        sim_context = sim["context"].lower()
                        query_context = context.lower()

                        # Simple keyword overlap check
                        sim_words = set(re.findall(r'\b\w+\b', sim_context))
                        query_words = set(re.findall(r'\b\w+\b', query_context))
                        overlap = len(sim_words & query_words) / max(len(query_words), 1)

                        if overlap < 0.3:  # At least 30% word overlap
                            continue

                    if related_kw not in expanded:
                        next_level.add(related_kw)
                        expanded.add(related_kw)
                        keyword_origin[related_kw] = origin

                        # Add to expansion map
                        if origin in expansion_map:
                            expansion_map[origin].append(related_kw)

            current_level = next_level
            if not current_level:
                break  # No more expansions found

        return list(expanded), expansion_map

    def search_with_keywords(
        self,
        question: str,
        keywords: List[str],
        context: str,
        threshold: float = 0.7,
        expand_depth: int = 1
    ) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Search documents by keywords and filter by LLM relevance.

        Args:
            question: User's question
            keywords: List of keywords to search
            context: User's context/domain
            threshold: Minimum relevance score (0.0-1.0)
            expand_depth: Number of levels to expand keywords (default: 1)

        Returns:
            Tuple of (scored_results, expansion_map)
            - scored_results: List of relevant documents with scores
            - expansion_map: Dict mapping original keywords to their expansions
        """
        # Expand keywords using similarities
        expanded_keywords, expansion_map = self.expand_keywords(
            keywords=keywords,
            context=context,
            depth=expand_depth
        )

        # Create reverse map: expanded_keyword -> original_keyword
        reverse_map: Dict[str, str] = {}
        normalized_originals = [self.db._normalize_keyword(kw) for kw in keywords]
        for orig_kw in normalized_originals:
            reverse_map[orig_kw] = orig_kw  # Original maps to itself
            for expanded_kw in expansion_map.get(orig_kw, []):
                reverse_map[expanded_kw] = orig_kw

        # Search by expanded keywords (OR mode)
        results = self.search_engine.search_by_keywords_or(expanded_keywords)

        # Score each document and track expansion
        scored_results = []
        for doc in results:
            doc_keywords = self.db.get_document_keywords(doc["filepath"])
            keyword_list = [kw["keyword"] for kw in doc_keywords]

            is_relevant, score, reasoning = self.score_document_relevance(
                question=question,
                context=context,
                doc_title=doc.get("title", ""),
                doc_summary=doc.get("summary", ""),
                doc_keywords=keyword_list
            )

            if is_relevant and score >= threshold:
                # Map matched keywords back to original keywords
                matched = doc.get("matched_keywords", [])
                user_keywords = []
                keyword_expansions = []

                for matched_kw in matched:
                    orig_kw = reverse_map.get(matched_kw, matched_kw)
                    user_keywords.append(orig_kw)

                    # Check if this was an expansion
                    if matched_kw != orig_kw:
                        keyword_expansions.append({
                            "original": orig_kw,
                            "expanded": matched_kw
                        })

                scored_results.append({
                    "filepath": doc["filepath"],
                    "title": doc.get("title"),
                    "summary": doc.get("summary"),
                    "matched_keywords": matched,
                    "user_keywords": user_keywords,
                    "keyword_expansions": keyword_expansions,
                    "relevance_score": score,
                    "reasoning": reasoning,
                    "source": "keyword_search"
                })

        # Sort by relevance score (highest first)
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return scored_results, expansion_map

    def grep_search(
        self,
        question: str,
        context: str,
        query_keywords: List[str],
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Fallback search using grep when keyword search fails.

        Args:
            question: User's question
            context: User's context/domain
            query_keywords: Original query keywords (to add to auto-indexed documents)
            threshold: Minimum relevance score (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            List of relevant documents found via grep
        """
        # Extract search terms from question using LLM
        terms = self._extract_search_terms(question, context)

        if not terms:
            return []

        # Find all markdown files in knowledge base
        if not self.knowledge_base_path.exists():
            return []

        # Use grep to search for terms
        all_matches = set()
        for term in terms:
            try:
                # Use grep -l to list files containing the term
                result = subprocess.run(
                    ['grep', '-ril', term, str(self.knowledge_base_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    matches = result.stdout.strip().split('\n')
                    all_matches.update([m for m in matches if m.endswith('.md')])
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue

        # Score each file
        scored_results = []
        for filepath in all_matches:
            # Convert to relative path
            try:
                rel_path = Path(filepath).relative_to(self.knowledge_base_path)
            except ValueError:
                continue

            # Check if document is indexed
            doc = self.db.get_document(str(rel_path))
            was_indexed = bool(doc)
            was_reindexed = False

            if doc:
                # Already indexed - check if needs reindexing
                reindex_result = self.reindex_document_if_modified(
                    filepath=str(rel_path),
                    query_keywords=query_keywords
                )

                if reindex_result["status"] == "reindexed":
                    was_reindexed = True
                    # Get updated document info
                    doc = self.db.get_document(str(rel_path))

                # Get current keywords (possibly updated)
                doc_keywords = self.db.get_document_keywords(str(rel_path))
                keyword_list = [kw["keyword"] for kw in doc_keywords]
                title = doc.get("title", "")
                summary = doc.get("summary", "")
            else:
                # Not indexed - auto-index it with query keywords
                if self.auto_index_document(str(rel_path), query_keywords=query_keywords):
                    # Successfully indexed, get the info
                    doc = self.db.get_document(str(rel_path))
                    doc_keywords = self.db.get_document_keywords(str(rel_path))
                    keyword_list = [kw["keyword"] for kw in doc_keywords]
                    title = doc.get("title", "")
                    summary = doc.get("summary", "")
                    was_indexed = False  # Mark as newly indexed
                else:
                    # Failed to index, skip this document
                    continue

            is_relevant, score, reasoning = self.score_document_relevance(
                question=question,
                context=context,
                doc_title=title,
                doc_summary=summary,
                doc_keywords=keyword_list
            )

            if is_relevant and score >= threshold:
                scored_results.append({
                    "filepath": str(rel_path),
                    "title": title,
                    "summary": summary,
                    "matched_keywords": keyword_list,
                    "relevance_score": score,
                    "reasoning": reasoning,
                    "source": "grep_search",
                    "indexed": True,  # Now indexed (either was before or just indexed)
                    "auto_indexed": not was_indexed,  # Flag if it was just auto-indexed
                    "reindexed": was_reindexed  # Flag if keywords were updated
                })

        # Sort by relevance score and limit results
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_results[:max_results]

    def _extract_search_terms(self, question: str, context: str) -> List[str]:
        """
        Extract key search terms from question using LLM.

        Args:
            question: User's question
            context: User's context

        Returns:
            List of search terms
        """
        prompt = f"""Extract 3-5 key search terms from this question that would be good for grep searching.

Question: {question}
Context: {context}

Return ONLY the search terms, one per line, without explanations.
Focus on distinctive technical terms, concepts, or phrases.

Your answer:"""

        try:
            answer = self._call_llm(prompt)
            terms = [line.strip() for line in answer.split('\n') if line.strip()]
            # Remove common words and duplicates
            terms = [t for t in terms if len(t) > 2]
            return list(set(terms))[:5]
        except Exception as e:
            print(f"Warning: Term extraction failed: {e}")
            # Fallback: extract words from question
            words = re.findall(r'\b[a-zA-Z]{4,}\b', question)
            return list(set(words))[:5]

    def _extract_title_from_file(self, filepath: str) -> str:
        """
        Extract title from markdown file.

        Args:
            filepath: Path to markdown file

        Returns:
            Title or filename
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('# '):
                        return line[2:].strip()
            return Path(filepath).stem
        except Exception:
            return Path(filepath).stem

    def _extract_summary_from_file(self, filepath: str) -> str:
        """
        Extract summary from markdown file (first paragraph).

        Args:
            filepath: Path to markdown file

        Returns:
            Summary or empty string
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read(500)  # Read first 500 chars
                # Find first paragraph after title
                lines = content.split('\n')
                summary = []
                found_content = False
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    found_content = True
                    summary.append(line)
                    if len(' '.join(summary)) > 200:
                        break
                return ' '.join(summary)[:200]
        except Exception:
            return ""

    def _needs_reindexing(self, filepath: str, doc: Dict) -> bool:
        """
        Check if a document needs reindexing based on file modification time.

        Args:
            filepath: Relative path to the document (e.g., "doc.md")
            doc: Document dictionary from database with 'updated_at' field

        Returns:
            True if file is newer than database timestamp, False otherwise
        """
        try:
            # Get absolute path
            abs_path = self.knowledge_base_path / filepath

            # Get file modification time
            file_mtime = abs_path.stat().st_mtime

            # Parse database timestamp (format: "YYYY-MM-DD HH:MM:SS")
            db_updated_at = doc.get("updated_at")
            if not db_updated_at:
                return True  # No timestamp, reindex

            # Parse datetime string (SQLite CURRENT_TIMESTAMP is in UTC)
            try:
                db_datetime = datetime.strptime(db_updated_at, "%Y-%m-%d %H:%M:%S")
                # Treat as UTC and convert to timestamp
                db_datetime_utc = db_datetime.replace(tzinfo=timezone.utc)
                db_timestamp = db_datetime_utc.timestamp()
            except ValueError:
                # Try with microseconds
                db_datetime = datetime.strptime(db_updated_at, "%Y-%m-%d %H:%M:%S.%f")
                db_datetime_utc = db_datetime.replace(tzinfo=timezone.utc)
                db_timestamp = db_datetime_utc.timestamp()

            # File is newer if its mtime is greater than db timestamp
            return file_mtime > db_timestamp

        except Exception as e:
            print(f"Warning: Could not check if {filepath} needs reindexing: {e}")
            return False

    def auto_index_document(self, filepath: str, query_keywords: Optional[List[str]] = None) -> bool:
        """
        Automatically index an unindexed document by generating keywords.

        Args:
            filepath: Relative path to the document (e.g., "doc.md")
            query_keywords: Optional query keywords to include in the index

        Returns:
            True if successfully indexed, False otherwise
        """
        try:
            # Get absolute path
            abs_path = self.knowledge_base_path / filepath

            # Read document content
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read(2000)  # First 2000 chars
            except Exception:
                return False

            # Generate title, summary, and keywords using LLM (same approach as generate_keywords.py)
            query_kw_section = ""
            if query_keywords:
                query_kw_section = f"""
QUERY CONTEXT:
These query keywords led to finding this document: {', '.join(query_keywords)}
Consider including relevant variations or related terms in your keyword extraction.

"""

            prompt = f"""You are a metadata extraction assistant. Analyze this markdown document and extract metadata.

{query_kw_section}TASK: Generate metadata following this process:
1. FIRST, think about 5-10 questions that people might ask that this document can answer
   - Questions should be natural, as users would phrase them
   - Think about what problems or needs would lead someone to this document

2. THEN, generate keywords based on what terms users asking those questions might search for
   - Extract 10-20 keywords total
   - Include terms from the questions and document content
   - Include both specific terms and general concepts
   - Use lowercase for keywords unless they are proper nouns or abbreviations

3. Generate title and summary:
   - Title: Extract from content (if document has # Title header) or create descriptive title in Title Case
   - Summary: Write 2-3 sentence summary of the document's main points

Return a JSON object with this structure:
{{
  "title": "Document Title",
  "summary": "2-3 sentence summary",
  "keywords": ["keyword1", "keyword2", ...]
}}

DOCUMENT CONTENT:
{content}

Your answer (JSON only):"""

            try:
                answer = self._call_llm(prompt)
                # Extract JSON object from response
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group(0))
                    title = metadata.get("title", "")
                    summary = metadata.get("summary", "")
                    keywords = metadata.get("keywords", [])
                else:
                    # LLM didn't return valid JSON - fail
                    print(f"Warning: LLM returned invalid JSON for {filepath}")
                    return False
            except Exception as e:
                # LLM call failed - fail
                print(f"Warning: LLM metadata generation failed for {filepath}: {e}")
                return False

            if not title or not keywords:
                return False

            # Add document to database
            self.db.add_document(
                filepath=filepath,
                title=title,
                summary=summary
            )

            # Add generated keywords
            for keyword in keywords:
                self.db.add_document_keyword(filepath, keyword)

            # Add query keywords if provided (these led to finding the document)
            if query_keywords:
                for kw in query_keywords:
                    normalized = self.db._normalize_keyword(kw)
                    # Only add if not already in keywords
                    if normalized not in [self.db._normalize_keyword(k) for k in keywords]:
                        self.db.add_document_keyword(filepath, kw)

            return True

        except Exception as e:
            print(f"Warning: Auto-indexing failed for {filepath}: {e}")
            return False

    def reindex_document_if_modified(
        self,
        filepath: str,
        query_keywords: Optional[List[str]] = None
    ) -> Dict:
        """
        Reindex a document if it has been modified since last indexing.
        Intelligently merges existing keywords with newly generated ones.

        Args:
            filepath: Relative path to the document (e.g., "doc.md")
            query_keywords: Optional query keywords to consider when reindexing

        Returns:
            Dictionary with reindexing status:
            - "status": "not_indexed", "up_to_date", "reindexed", or "error"
            - "keywords_before": List of keywords before reindexing (if applicable)
            - "keywords_after": List of keywords after reindexing (if applicable)
            - "added": List of keywords added
            - "removed": List of keywords removed
            - "kept": List of keywords kept
        """
        try:
            # Check if document is indexed
            doc = self.db.get_document(filepath)
            if not doc:
                return {
                    "status": "not_indexed",
                    "message": "Document not in database"
                }

            # Check if needs reindexing
            if not self._needs_reindexing(filepath, doc):
                return {
                    "status": "up_to_date",
                    "message": "Document is up to date"
                }

            # Get existing keywords
            existing_kw_data = self.db.get_document_keywords(filepath)
            existing_keywords = [kw["keyword"] for kw in existing_kw_data]

            # Get absolute path
            abs_path = self.knowledge_base_path / filepath

            # Read document content
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read(2000)  # First 2000 chars
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Could not read file: {e}"
                }

            # Generate updated metadata using LLM
            query_kw_section = ""
            if query_keywords:
                query_kw_section = f"""
QUERY CONTEXT:
These query keywords led to finding this document: {', '.join(query_keywords)}
Consider including relevant variations or related terms.

"""

            prompt = f"""You are a metadata extraction assistant. Analyze this updated document and intelligently update its metadata.

EXISTING KEYWORDS:
{', '.join(existing_keywords)}

{query_kw_section}TASK: Generate updated metadata following this process:
1. FIRST, think about 5-10 questions that people might ask that this document can answer
   - Questions should be natural, as users would phrase them
   - Think about what problems or needs would lead someone to this document

2. THEN, decide which keywords to keep, add, or remove:
   - KEEP: Existing keywords that are still relevant to the document
   - ADD: New keywords based on what terms users asking those questions might search for
   - REMOVE: Existing keywords that are no longer relevant

3. Generate updated title and summary:
   - Title: Extract from content (if document has # Title header) or create descriptive title in Title Case
   - Summary: Write 2-3 sentence summary of the document's main points

Guidelines for keywords:
- Extract 10-20 keywords total (after keep/add/remove)
- Include terms from potential questions and document content
- Include both specific terms and general concepts
- Use lowercase for keywords unless they are proper nouns or abbreviations
- Be conservative with removals - only remove if clearly no longer relevant

Return a JSON object with this structure:
{{
  "title": "Document Title",
  "summary": "2-3 sentence summary",
  "keep": ["keyword1", "keyword2"],     // Existing keywords to keep
  "add": ["keyword3", "keyword4"],      // New keywords to add
  "remove": ["keyword5"]                // Existing keywords to remove
}}

DOCUMENT CONTENT:
{content}

Your answer (JSON only):"""

            try:
                answer = self._call_llm(prompt)
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group(0))
                    title = metadata.get("title", "")
                    summary = metadata.get("summary", "")
                    keep = metadata.get("keep", [])
                    add = metadata.get("add", [])
                    remove = metadata.get("remove", [])
                else:
                    # Fallback: keep all existing, add query keywords
                    title = ""
                    summary = ""
                    keep = existing_keywords
                    add = query_keywords or []
                    remove = []
            except Exception as e:
                print(f"Warning: LLM metadata generation failed: {e}")
                # Fallback: keep all existing
                title = ""
                summary = ""
                keep = existing_keywords
                add = query_keywords or []
                remove = []

            # Compute final keyword list
            final_keywords = list(set(keep + add))

            # Remove keywords marked for removal
            final_keywords = [kw for kw in final_keywords if kw not in remove]

            # Update document title and summary (only if AI provided them)
            if title and summary:
                self.db.update_document(filepath, title=title, summary=summary)

            # Replace keywords in database
            keywords_data = [(kw, None) for kw in final_keywords]
            self.db.replace_document_keywords(filepath, keywords_data)

            return {
                "status": "reindexed",
                "keywords_before": existing_keywords,
                "keywords_after": final_keywords,
                "added": [kw for kw in add if kw in final_keywords],
                "removed": remove,
                "kept": [kw for kw in keep if kw in final_keywords]
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Reindexing failed: {e}"
            }

    def generate_learning_suggestions(
        self,
        question: str,
        context: str,
        results: List[Dict]
    ) -> Dict:
        """
        Generate keyword and similarity suggestions based on query results.

        Args:
            question: User's question
            context: User's context
            results: List of query results

        Returns:
            Dictionary with keyword and similarity suggestions
        """
        if not results:
            return {"keyword_suggestions": [], "similarity_suggestions": []}

        # Prepare document information for LLM
        docs_info = []
        for result in results[:3]:  # Analyze top 3 results
            docs_info.append({
                "filepath": result["filepath"],
                "title": result.get("title", ""),
                "keywords": result.get("matched_keywords", []),
                "score": result["relevance_score"]
            })

        prompt = f"""Analyze these query results and suggest improvements to the keyword index.

User's Question: {question}
User's Context: {context}

Retrieved Documents:
{json.dumps(docs_info, indent=2)}

Task: Suggest new keywords and keyword relationships that would improve future searches.

Provide suggestions in JSON format:
{{
  "keyword_suggestions": [
    {{
      "filepath": "doc.md",
      "keywords": ["keyword1", "keyword2"],
      "reasoning": "Why these keywords help"
    }}
  ],
  "similarity_suggestions": [
    {{
      "keyword1": "term1",
      "keyword2": "term2",
      "type": "synonym|abbreviation|related",
      "context": "domain context",
      "score": 0.8,
      "reasoning": "Why this relationship helps"
    }}
  ]
}}

Focus on:
1. Missing keywords that appear in the question but not in document keywords
2. Relationships between question terms and existing keywords
3. Only suggest high-quality, useful additions

Your answer (JSON only):"""

        try:
            answer = self._call_llm(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', answer, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(0))
                return suggestions
            return {"keyword_suggestions": [], "similarity_suggestions": []}
        except Exception as e:
            print(f"Warning: Learning suggestion generation failed: {e}")
            return {"keyword_suggestions": [], "similarity_suggestions": []}

    def _generate_query_based_suggestions(
        self,
        question: str,
        context: str,
        query_keywords: List[str],
        results: List[Dict]
    ) -> Dict:
        """
        Generate learning suggestions focused on query keywords.

        For documents found via grep:
        - Add query keywords directly to document keywords
        - OR add query keywords as similarities to existing document keywords

        Args:
            question: User's question
            context: User's context
            query_keywords: Keywords from the query (--keywords)
            results: Grep search results

        Returns:
            Dictionary with keyword_suggestions and similarity_suggestions
        """
        keyword_suggestions = []
        similarity_suggestions = []

        for result in results:
            filepath = result["filepath"]
            existing_keywords = result.get("matched_keywords", [])

            # For each query keyword, decide if it should be:
            # 1. Added directly to the document
            # 2. Added as a similarity to an existing keyword

            for query_kw in query_keywords:
                normalized_query = self.db._normalize_keyword(query_kw)

                # Check if already in document keywords
                existing_normalized = [self.db._normalize_keyword(k) for k in existing_keywords]
                if normalized_query in existing_normalized:
                    continue  # Already has this keyword

                # Decide: add as keyword or as similarity?
                if existing_keywords:
                    # Document has keywords - add as similarity to the most relevant one
                    # Use the first existing keyword as the anchor
                    anchor_keyword = existing_keywords[0]

                    # Check if similarity already exists
                    existing_sims = self.db.get_similar_keywords(anchor_keyword, similarity_type=None)
                    sim_exists = any(
                        s["related_keyword"] == normalized_query
                        for s in existing_sims
                    )

                    if not sim_exists:
                        similarity_suggestions.append({
                            "keyword1": anchor_keyword,
                            "keyword2": query_kw,
                            "type": "related",
                            "context": context,
                            "score": 0.7,
                            "reasoning": f"Query keyword '{query_kw}' led to finding document '{filepath}'"
                        })
                else:
                    # No existing keywords (shouldn't happen with auto-index, but handle it)
                    # Add as document keyword
                    keyword_suggestions.append({
                        "filepath": filepath,
                        "keywords": [query_kw],
                        "reasoning": f"Query keyword '{query_kw}' led to finding this document"
                    })

        return {
            "keyword_suggestions": keyword_suggestions,
            "similarity_suggestions": similarity_suggestions
        }

    def apply_learning_suggestions(self, suggestions: Dict) -> Dict:
        """
        Apply learning suggestions to the database.

        Args:
            suggestions: Dictionary with keyword_suggestions and similarity_suggestions

        Returns:
            Dictionary with counts of applied suggestions
        """
        applied = {
            "keywords_added": 0,
            "similarities_added": 0,
            "errors": []
        }

        # Apply keyword suggestions
        for kw_sugg in suggestions.get("keyword_suggestions", []):
            try:
                filepath = kw_sugg["filepath"]
                keywords = kw_sugg["keywords"]

                # Check if document exists
                doc = self.db.get_document(filepath)
                if not doc:
                    applied["errors"].append(
                        f"Document not indexed: {filepath} - run './kbindex.py add {filepath}' to index it first"
                    )
                    continue

                # Add each keyword
                for keyword in keywords:
                    # Check if keyword already exists for this document
                    existing_keywords = self.db.get_document_keywords(filepath)
                    existing_kw_set = {kw["keyword"] for kw in existing_keywords}

                    normalized = self.db._normalize_keyword(keyword)
                    if normalized not in existing_kw_set:
                        self.db.add_document_keyword(filepath, keyword)
                        applied["keywords_added"] += 1

            except Exception as e:
                applied["errors"].append(f"Error adding keywords to {kw_sugg.get('filepath', 'unknown')}: {str(e)}")

        # Apply similarity suggestions
        for sim_sugg in suggestions.get("similarity_suggestions", []):
            try:
                keyword1 = sim_sugg["keyword1"]
                keyword2 = sim_sugg["keyword2"]
                sim_type = sim_sugg["type"]
                sim_context = sim_sugg["context"]
                score = sim_sugg.get("score", 0.5)

                # Check if similarity already exists
                existing_sims = self.db.get_similar_keywords(keyword1, sim_type)
                exists = any(
                    s["related_keyword"] == self.db._normalize_keyword(keyword2)
                    for s in existing_sims
                )

                if not exists:
                    self.db.add_similarity(
                        keyword1=keyword1,
                        keyword2=keyword2,
                        similarity_type=sim_type,
                        context=sim_context,
                        score=score,
                        directional=False
                    )
                    applied["similarities_added"] += 1

            except Exception as e:
                applied["errors"].append(
                    f"Error adding similarity {sim_sugg.get('keyword1', '?')} ↔ {sim_sugg.get('keyword2', '?')}: {str(e)}"
                )

        return applied

    def query(
        self,
        question: str,
        keywords: List[str],
        context: str,
        threshold: float = 0.7,
        expand_depth: int = 1,
        enable_grep_fallback: bool = True,
        enable_learning: bool = True,
        auto_apply: bool = True
    ) -> Dict:
        """
        Intelligent query combining keyword search, LLM filtering, and learning.

        Args:
            question: User's question
            keywords: Initial keywords to search
            context: User's context/domain
            threshold: Minimum relevance score (default: 0.7)
            expand_depth: Number of levels to expand keywords using similarities (default: 1)
            enable_grep_fallback: Enable grep search if keyword search fails
            enable_learning: Generate and optionally apply learning suggestions
            auto_apply: Automatically apply learning suggestions to database (default: True)

        Returns:
            Query results with documents, suggestions, and metadata
        """
        # Phase 1: Keyword search with expansion
        keyword_results, expansion_map = self.search_with_keywords(
            question=question,
            keywords=keywords,
            context=context,
            threshold=threshold,
            expand_depth=expand_depth
        )

        # Get expanded keywords from expansion map
        expanded_keywords = list(set(keywords))
        if expand_depth > 0:
            for orig_kw in expansion_map:
                expanded_keywords.extend(expansion_map[orig_kw])

        # Phase 2: Grep fallback if needed
        grep_results = []
        if enable_grep_fallback and len(keyword_results) == 0:
            grep_results = self.grep_search(
                question=question,
                context=context,
                query_keywords=keywords,
                threshold=threshold
            )

        # Combine results
        all_results = keyword_results + grep_results

        # Phase 3: Learning from grep results
        # ONLY learn if keyword search (with expansion) FAILED but grep found documents
        # This means the index needs improvement
        suggestions = {}
        applied = None
        if enable_learning and len(keyword_results) == 0 and len(grep_results) > 0:
            # Keyword search failed, but grep found relevant documents
            # Add query keywords to these documents or as similarities
            indexed_results = [r for r in grep_results if r.get("indexed", True)]

            if indexed_results:
                # Generate suggestions based on query keywords and grep results
                suggestions = self._generate_query_based_suggestions(
                    question=question,
                    context=context,
                    query_keywords=keywords,
                    results=indexed_results
                )

                # Phase 4: Auto-apply suggestions
                if auto_apply and suggestions:
                    applied = self.apply_learning_suggestions(suggestions)

        return {
            "query": {
                "question": question,
                "keywords": keywords,
                "expanded_keywords": expanded_keywords if expand_depth > 0 else None,
                "expansion_map": expansion_map if expand_depth > 0 else None,
                "context": context,
                "threshold": threshold,
                "expand_depth": expand_depth
            },
            "results": all_results,
            "count": len(all_results),
            "keyword_search_count": len(keyword_results),
            "grep_search_count": len(grep_results),
            "suggestions": suggestions if enable_learning else None,
            "applied": applied if (enable_learning and auto_apply) else None
        }
