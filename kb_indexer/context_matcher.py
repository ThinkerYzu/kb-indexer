#!/usr/bin/env python3
"""
Context Matcher - AI-powered context matching for keyword similarities.

This module provides LLM-based evaluation of whether a keyword similarity
relationship is relevant to a user's provided context.

Supports both cloud (Gemini) and local (Ollama) LLM backends.
"""

import os
import re
from typing import Tuple, Optional, Literal


class ContextMatcher:
    """
    Evaluates if similarity contexts match user-provided context using LLM.
    """

    def __init__(
        self,
        backend: Literal["gemini", "ollama"] = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the ContextMatcher with specified LLM backend.

        Args:
            backend: LLM backend to use ("gemini" or "ollama")
            api_key: API key for Gemini. If None, reads from GEMINI_API_KEY env variable.
            model: Model name. Defaults: gemini-2.0-flash-exp for Gemini, llama3.2:3b for Ollama.

        Raises:
            ValueError: If Gemini backend selected but no API key provided.
            ImportError: If required backend library is not installed.
        """
        self.backend = backend

        if backend == "gemini":
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
            raise ValueError(f"Unknown backend: {backend}. Use 'gemini' or 'ollama'.")

    def matches(
        self,
        keyword: str,
        related_keyword: str,
        similarity_type: str,
        similarity_context: str,
        user_context: str
    ) -> Tuple[bool, float]:
        """
        Evaluate if a similarity is relevant to the user's context.

        Args:
            keyword: The original keyword being queried
            related_keyword: The similar/related keyword
            similarity_type: Type of similarity (e.g., 'abbreviation', 'application')
            similarity_context: The stored context describing the similarity
            user_context: The user's context/domain for filtering

        Returns:
            Tuple of (matches: bool, confidence_score: float)
            - matches: True if similarity is relevant to user context
            - confidence_score: 0.0-1.0 confidence in the decision

        Example:
            >>> matcher = ContextMatcher()
            >>> matcher.matches(
            ...     'RL',
            ...     'AlphaGo',
            ...     'application',
            ...     'AlphaGo demonstrates RL in game of Go',
            ...     'game AI and competitions'
            ... )
            (True, 0.92)
        """
        prompt = f"""Evaluate if this keyword similarity is relevant to the user's context.

Original Keyword: {keyword}
Related Keyword: {related_keyword}
Similarity Type: {similarity_type}
Similarity Context: {similarity_context}
User's Context: {user_context}

Task: Evaluate if this similarity relationship is relevant to the user's context.

Similarity Type Priority (when contexts are equally relevant):
1. synonym, abbreviation - Fundamental understanding (highest priority)
2. related_concept - Core concepts in same domain
3. broader, narrower - Hierarchical relationships
4. component, prerequisite - Structural relationships
5. application - Specific examples/implementations (lower priority)
6. contrast - Opposing concepts (usually irrelevant, answer "no")

Scoring Guidelines:
- 0.95-1.0: Highest priority type + perfect context match
- 0.85-0.94: High priority type + strong context match, OR lower priority + perfect match
- 0.75-0.84: Medium priority type + good context match
- 0.65-0.74: Lower priority type + moderate context match
- Below 0.65: Context mismatch (answer "no")

Evaluation Process:
1. Check if similarity context domain aligns with user's context domain
2. Consider the similarity type priority
3. Combine both factors to determine the score

Key Principle: Fundamental relationships (synonyms, abbreviations) are generally more valuable than specific examples (applications), even if the example domain matches better.

Examples:
- Type: abbreviation, Context: "ML literature", User: "ML research" → yes,0.95 (high priority + perfect match)
- Type: abbreviation, Context: "ML literature", User: "game AI" → yes,0.85 (high priority + related domain)
- Type: application, Context: "game of Go", User: "game AI" → yes,0.80 (lower priority + perfect match)
- Type: application, Context: "game of Go", User: "robotics" → yes,0.65 (lower priority + weak match)

Answer ONLY in format: yes|no,score
Your answer:"""

        try:
            if self.backend == "gemini":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                answer = response.text.strip().lower()

            elif self.backend == "ollama":
                response = self.ollama.generate(
                    model=self.model,
                    prompt=prompt
                )
                answer = response['response'].strip().lower()

            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            # Extract decision and score using regex
            match = re.match(r'(yes|no),(\d+\.?\d*)', answer)

            if not match:
                # Fallback parsing if format is slightly different
                if 'yes' in answer:
                    decision = True
                    # Try to extract any number
                    score_match = re.search(r'(\d+\.?\d*)', answer)
                    score = float(score_match.group(1)) if score_match else 0.5
                elif 'no' in answer:
                    decision = False
                    score_match = re.search(r'(\d+\.?\d*)', answer)
                    score = float(score_match.group(1)) if score_match else 0.5
                else:
                    # Unable to parse, default to no match with low confidence
                    return (False, 0.3)
            else:
                decision = match.group(1) == 'yes'
                score = float(match.group(2))

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))

            return (decision, score)

        except Exception as e:
            # On error, return no match with low confidence
            print(f"Warning: Context matching failed: {e}")
            return (False, 0.0)
