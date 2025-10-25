"""
Search module for Knowledge Base Indexer.

Provides search functionality for documents by keywords.
"""

from typing import Dict, List, Optional, Set

from .database import Database


class SearchEngine:
    """Search engine for knowledge base documents."""

    def __init__(self, db: Database):
        """Initialize search engine.

        Args:
            db: Database instance
        """
        self.db = db

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Search documents by single keyword.

        Args:
            keyword: Keyword to search for

        Returns:
            List of matching documents with matched keywords
        """
        docs = self.db.get_documents_by_keyword(keyword)

        # Add matched_keywords field - show the user-given keyword that found the doc
        normalized_query = self.db._normalize_keyword(keyword)
        for doc in docs:
            doc["matched_keywords"] = [normalized_query]
            doc["user_keywords"] = [keyword]  # Original user-provided keyword

        return docs

    def search_by_keywords_or(self, keywords: List[str]) -> List[Dict]:
        """Search documents matching ANY of the keywords (OR).

        Args:
            keywords: List of keywords to search for

        Returns:
            List of matching documents with matched keywords
        """
        # Use set to track unique documents
        doc_map: Dict[str, Dict] = {}

        for keyword in keywords:
            docs = self.db.get_documents_by_keyword(keyword)
            for doc in docs:
                filepath = doc["filepath"]
                if filepath not in doc_map:
                    doc_map[filepath] = doc
                    doc_map[filepath]["matched_keywords"] = []
                    doc_map[filepath]["user_keywords"] = []

        # Add matched keywords for each document
        for filepath, doc in doc_map.items():
            doc_keywords = self.db.get_document_keywords(filepath)
            doc_kw_set = {kw["keyword"] for kw in doc_keywords}

            # Find which query keywords matched
            matched = []
            user_kws = []
            for query_kw in keywords:
                normalized = self.db._normalize_keyword(query_kw)
                if normalized in doc_kw_set:
                    matched.append(normalized)
                    user_kws.append(query_kw)

            doc["matched_keywords"] = matched
            doc["user_keywords"] = user_kws

        # Return sorted by filepath
        return sorted(doc_map.values(), key=lambda d: d["filepath"])

    def search_by_keywords_and(self, keywords: List[str]) -> List[Dict]:
        """Search documents matching ALL of the keywords (AND).

        Args:
            keywords: List of keywords to search for

        Returns:
            List of matching documents with matched keywords
        """
        if not keywords:
            return []

        # Normalize all query keywords
        normalized_keywords = [self.db._normalize_keyword(kw) for kw in keywords]

        # Get documents for first keyword
        first_keyword = normalized_keywords[0]
        docs = self.db.get_documents_by_keyword(first_keyword)

        # Filter to only documents that have ALL keywords
        matching_docs = []

        for doc in docs:
            doc_keywords = self.db.get_document_keywords(doc["filepath"])
            doc_kw_set = {kw["keyword"] for kw in doc_keywords}

            # Check if all query keywords are in this document
            if all(kw in doc_kw_set for kw in normalized_keywords):
                doc["matched_keywords"] = normalized_keywords
                doc["user_keywords"] = keywords  # Original user-provided keywords
                matching_docs.append(doc)

        return matching_docs

    def get_similar_keywords_for_search(
        self,
        keyword: str,
        similarity_type: Optional[str] = None,
    ) -> List[str]:
        """Get list of similar keywords suitable for search expansion.

        Args:
            keyword: Keyword to find similarities for
            similarity_type: Optional filter by similarity type

        Returns:
            List of similar keyword strings (not including original)
        """
        similarities = self.db.get_similar_keywords(keyword, similarity_type)
        return [sim["related_keyword"] for sim in similarities]

    def search_with_expansion(
        self,
        keywords: List[str],
        expand_synonyms: bool = True,
        expand_abbreviations: bool = True,
        expand_related: bool = False,
        similarity_types: Optional[List[str]] = None,
    ) -> Dict:
        """Search with automatic keyword expansion based on similarities.

        This is a convenience method. AI agents should typically perform
        expansion themselves after querying similarities with context.

        Args:
            keywords: Original search keywords
            expand_synonyms: Include 'synonym' type similarities
            expand_abbreviations: Include 'abbreviation' type similarities
            expand_related: Include 'related_concept' type similarities
            similarity_types: Custom list of similarity types to include

        Returns:
            Dictionary with expanded_keywords and search results
        """
        # Determine which similarity types to include
        if similarity_types is None:
            similarity_types = []
            if expand_synonyms:
                similarity_types.append("synonym")
            if expand_abbreviations:
                similarity_types.append("abbreviation")
            if expand_related:
                similarity_types.append("related_concept")

        # Collect expanded keywords
        expanded_keywords = set(keywords)

        for keyword in keywords:
            for sim_type in similarity_types:
                similar = self.get_similar_keywords_for_search(keyword, sim_type)
                expanded_keywords.update(similar)

        # Perform OR search with expanded keywords
        results = self.search_by_keywords_or(list(expanded_keywords))

        return {
            "original_keywords": keywords,
            "expanded_keywords": sorted(expanded_keywords),
            "expansion_types": similarity_types,
            "results": results,
            "count": len(results),
        }

    def format_search_results(
        self,
        results: List[Dict],
        query_keywords: List[str],
        mode: str = "exact",
    ) -> Dict:
        """Format search results for JSON output.

        Args:
            results: List of document dictionaries
            query_keywords: Original query keywords
            mode: Search mode ('exact', 'or', 'and', 'expanded')

        Returns:
            Formatted result dictionary
        """
        return {
            "query": {
                "keywords": query_keywords,
                "mode": mode,
            },
            "results": [
                {
                    "filepath": doc["filepath"],
                    "title": doc.get("title"),
                    "summary": doc.get("summary"),
                    "matched_keywords": doc.get("matched_keywords", []),
                    "user_keywords": doc.get("user_keywords", []),
                }
                for doc in results
            ],
            "count": len(results),
        }

    def format_similar_keywords(self, keyword: str, similarities: List[Dict]) -> Dict:
        """Format similar keywords for JSON output.

        Args:
            keyword: Original keyword
            similarities: List of similarity dictionaries from database

        Returns:
            Formatted result dictionary
        """
        return {
            "keyword": keyword,
            "similar_keywords": [
                {
                    "keyword": sim["related_keyword"],
                    "similarity_type": sim["similarity_type"],
                    "context": sim["context"],
                    "score": sim["score"],
                    "directional": bool(sim["directional"]),
                    **({'context_match_score': sim['context_match_score']} if 'context_match_score' in sim else {})
                }
                for sim in similarities
            ],
            "count": len(similarities),
        }

    def format_document_details(self, filepath: str) -> Optional[Dict]:
        """Format document details for JSON output.

        Args:
            filepath: Document filepath

        Returns:
            Formatted document dictionary or None if not found
        """
        doc = self.db.get_document(filepath)
        if not doc:
            return None

        keywords = self.db.get_document_keywords(filepath)

        return {
            "filepath": doc["filepath"],
            "title": doc.get("title"),
            "summary": doc.get("summary"),
            "keywords": [
                {"keyword": kw["keyword"], "category": kw.get("category")}
                for kw in keywords
            ],
            "created_at": doc["created_at"],
            "updated_at": doc["updated_at"],
        }
