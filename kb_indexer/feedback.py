"""
Feedback Learning Engine - Analyzes user feedback to improve the knowledge base index.

This module processes user feedback from queries to identify patterns and
generate suggestions for improving keywords and similarities.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from .database import Database


class FeedbackLearner:
    """
    Analyzes user feedback to generate improvement suggestions.
    """

    def __init__(self, db: Database):
        """
        Initialize FeedbackLearner.

        Args:
            db: Database instance
        """
        self.db = db

    def analyze_feedback(self) -> Dict:
        """
        Analyze all unprocessed feedback and generate improvement suggestions.

        Returns:
            Dictionary with suggestions and metadata
        """
        # Get all unprocessed feedback
        feedback = self.db.get_unprocessed_feedback()

        if not feedback:
            return {
                "status": "no_feedback",
                "message": "No unprocessed feedback to analyze",
                "suggestions": [],
                "count": 0,
            }

        # Generate suggestions
        suggestions = {
            "keyword_gaps": self._find_keyword_gaps(feedback),
            "keyword_augmentation": self._find_keyword_augmentation(feedback),
            "pattern_recognition": self._find_pattern_recognition(feedback),
        }

        # Flatten suggestions
        all_suggestions = []
        all_suggestions.extend(suggestions["keyword_gaps"])
        all_suggestions.extend(suggestions["keyword_augmentation"])
        all_suggestions.extend(suggestions["pattern_recognition"])

        return {
            "status": "success",
            "feedback_count": len(feedback),
            "query_count": len(set(fb["query_id"] for fb in feedback)),
            "suggestions": all_suggestions,
            "breakdown": {
                "keyword_gaps": len(suggestions["keyword_gaps"]),
                "keyword_augmentation": len(suggestions["keyword_augmentation"]),
                "pattern_recognition": len(suggestions["pattern_recognition"]),
            },
            "feedback_ids": [fb["id"] for fb in feedback],
        }

    def _find_keyword_gaps(self, feedback: List[Dict]) -> List[Dict]:
        """
        Find common gaps between user search terms and document keywords.

        Args:
            feedback: List of feedback dictionaries

        Returns:
            List of similarity suggestion dictionaries
        """
        gap_counts = defaultdict(lambda: {"count": 0, "context": None})

        for fb in feedback:
            if not fb["keywords"]:
                continue

            query_keywords = fb["keywords"]
            doc_keywords = self.db.get_document_keywords(fb["filepath"])
            doc_kw_set = {kw["keyword"].lower() for kw in doc_keywords}

            for query_kw in query_keywords:
                normalized = self.db._normalize_keyword(query_kw)
                # Check if this keyword is missing from document
                if normalized not in doc_kw_set:
                    # Find similar keywords in doc
                    for doc_kw in doc_kw_set:
                        if doc_kw != normalized:
                            key = (normalized, doc_kw)
                            gap_counts[key]["count"] += 1
                            gap_counts[key]["context"] = fb["context"]

        # Convert to suggestions (only high-confidence ones)
        suggestions = []
        for (query_kw, doc_kw), data in gap_counts.items():
            if data["count"] >= 2:  # At least 2 occurrences
                # Determine similarity type based on pattern
                similarity_type = "related"
                score = min(0.5 + (data["count"] * 0.1), 0.9)  # Score increases with count

                # Check if looks like abbreviation
                if len(query_kw) < len(doc_kw) and len(query_kw) <= 5:
                    similarity_type = "abbreviation"
                    score = min(0.7 + (data["count"] * 0.1), 0.95)

                suggestions.append(
                    {
                        "type": "similarity",
                        "keyword1": query_kw,
                        "keyword2": doc_kw,
                        "similarity_type": similarity_type,
                        "context": data["context"],
                        "score": score,
                        "reason": f"User searched '{query_kw}' but helpful docs have '{doc_kw}' ({data['count']} occurrences)",
                        "confidence": data["count"],
                    }
                )

        return suggestions

    def _find_keyword_augmentation(self, feedback: List[Dict]) -> List[Dict]:
        """
        Find user keywords that consistently led to helpful documents.

        Args:
            feedback: List of feedback dictionaries

        Returns:
            List of keyword augmentation suggestions
        """
        doc_keyword_map = defaultdict(lambda: defaultdict(int))

        for fb in feedback:
            if not fb["keywords"]:
                continue

            filepath = fb["filepath"]
            for keyword in fb["keywords"]:
                normalized = self.db._normalize_keyword(keyword)
                doc_keyword_map[filepath][normalized] += 1

        # Convert to suggestions
        suggestions = []
        for filepath, keywords in doc_keyword_map.items():
            doc_kw = self.db.get_document_keywords(filepath)
            existing_kw_set = {self.db._normalize_keyword(kw["keyword"]) for kw in doc_kw}

            for keyword, count in keywords.items():
                if count >= 2 and keyword not in existing_kw_set:
                    suggestions.append(
                        {
                            "type": "keyword_augmentation",
                            "filepath": filepath,
                            "keyword": keyword,
                            "reason": f"User keyword led to finding this helpful document {count} times",
                            "confidence": count,
                        }
                    )

        return suggestions

    def _find_pattern_recognition(self, feedback: List[Dict]) -> List[Dict]:
        """
        Find keyword combinations that appear in successful queries.

        Args:
            feedback: List of feedback dictionaries

        Returns:
            List of pattern-based suggestions
        """
        keyword_pairs = defaultdict(int)
        query_keywords = defaultdict(set)

        for fb in feedback:
            if not fb["keywords"]:
                continue

            query_id = fb["query_id"]
            for keyword in fb["keywords"]:
                normalized = self.db._normalize_keyword(keyword)
                query_keywords[query_id].add(normalized)

        # Find co-occurring keywords
        for keywords in query_keywords.values():
            kw_list = sorted(list(keywords))
            for i in range(len(kw_list)):
                for j in range(i + 1, len(kw_list)):
                    pair = (kw_list[i], kw_list[j])
                    keyword_pairs[pair] += 1

        # Convert to suggestions (threshold: 2 co-occurrences)
        suggestions = []
        for (kw1, kw2), count in keyword_pairs.items():
            if count >= 2:
                # Check if similarity already exists
                existing = self.db.get_similar_keywords(kw1, None)
                exists = any(
                    sim["related_keyword"] == kw2 for sim in existing
                )

                if not exists:
                    suggestions.append(
                        {
                            "type": "similarity",
                            "keyword1": kw1,
                            "keyword2": kw2,
                            "similarity_type": "related",
                            "context": "Keywords co-occur in successful queries",
                            "score": min(0.5 + (count * 0.1), 0.7),
                            "reason": f"Keywords appear together in {count} successful queries",
                            "confidence": count,
                        }
                    )

        return suggestions

    def apply_suggestions(self, suggestions: List[Dict]) -> Dict:
        """
        Apply all suggestions to the database.

        Args:
            suggestions: List of suggestion dictionaries

        Returns:
            Dictionary with counts of applied suggestions
        """
        applied = {
            "similarities_added": 0,
            "keywords_added": 0,
            "errors": [],
        }

        for suggestion in suggestions:
            try:
                if suggestion["type"] == "similarity":
                    # Check if already exists
                    existing = self.db.get_similar_keywords(
                        suggestion["keyword1"], suggestion["similarity_type"]
                    )
                    exists = any(
                        sim["related_keyword"] == self.db._normalize_keyword(
                            suggestion["keyword2"]
                        )
                        for sim in existing
                    )

                    if not exists:
                        self.db.add_similarity(
                            keyword1=suggestion["keyword1"],
                            keyword2=suggestion["keyword2"],
                            similarity_type=suggestion["similarity_type"],
                            context=suggestion["context"],
                            score=suggestion["score"],
                            directional=False,
                        )
                        applied["similarities_added"] += 1

                elif suggestion["type"] == "keyword_augmentation":
                    # Check if keyword already exists for document
                    existing = self.db.get_document_keywords(suggestion["filepath"])
                    existing_kw_set = {
                        self.db._normalize_keyword(kw["keyword"]) for kw in existing
                    }

                    normalized = self.db._normalize_keyword(suggestion["keyword"])
                    if normalized not in existing_kw_set:
                        self.db.add_document_keyword(
                            suggestion["filepath"], suggestion["keyword"]
                        )
                        applied["keywords_added"] += 1

            except Exception as e:
                applied["errors"].append(f"Error applying suggestion: {str(e)}")

        return applied
