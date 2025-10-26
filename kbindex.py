#!/usr/bin/env python3
"""
Knowledge Base Indexer CLI

Command-line interface for indexing and searching markdown documents.
By default, looks for documents in ./knowledge-base/ directory.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from kb_indexer import Database, KeywordParser, MarkdownParser, SearchEngine, QueryEngine
from kb_indexer.parser import SimilarityParser


class CLI:
    """Command-line interface for kbindex."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize CLI.

        Args:
            db_path: Path to SQLite database (default: kb_index.db in script directory)
        """
        if db_path is None:
            # Default to kb_index.db in the same directory as kbindex.py
            script_dir = Path(__file__).parent
            db_path = str(script_dir / "kb_index.db")
        self.db_path = db_path
        self.db: Optional[Database] = None

    @staticmethod
    def utc_to_local(utc_time_str: str) -> str:
        """Convert UTC timestamp string to local time string.

        Args:
            utc_time_str: UTC timestamp in format 'YYYY-MM-DD HH:MM:SS'

        Returns:
            Local time string in same format
        """
        try:
            dt_utc = datetime.strptime(utc_time_str, '%Y-%m-%d %H:%M:%S')
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            dt_local = dt_utc.astimezone()
            return dt_local.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return utc_time_str  # Return original if conversion fails

    def _open_db(self):
        """Open database connection."""
        if not self.db:
            self.db = Database(self.db_path)

    def _close_db(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            self.db = None

    # ==================== Document Commands ====================

    def cmd_add(self, args):
        """Add document to index."""
        self._open_db()

        filepath = args.filepath
        keywords_file = args.keywords

        # If keywords file not specified, look for <filepath>.keywords.json
        if not keywords_file:
            kb_path = Path(filepath)
            keywords_file = kb_path.with_suffix(kb_path.suffix + ".keywords.json")

        # Parse keywords file
        try:
            kw_data = KeywordParser.parse_file(keywords_file)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing keywords file: {e}", file=sys.stderr)
            return 1

        # Extract title from markdown if not in keywords file
        title = kw_data.get("title")
        if not title:
            try:
                title = MarkdownParser.extract_title(filepath)
            except Exception:
                pass

        # Add document
        try:
            doc_id = self.db.add_document(
                filepath=kw_data["filepath"],
                title=title,
                summary=kw_data["summary"],
            )

            # Add keywords
            categories = kw_data.get("categories", {})
            for keyword in kw_data["keywords"]:
                # Find category for this keyword
                category = None
                for cat_name, cat_keywords in categories.items():
                    if keyword.lower() in [k.lower() for k in cat_keywords]:
                        category = cat_name
                        break

                self.db.add_document_keyword(kw_data["filepath"], keyword, category)

            print(f"Added document: {kw_data['filepath']}")
            return 0

        except Exception as e:
            print(f"Error adding document: {e}", file=sys.stderr)
            return 1

    def cmd_update(self, args):
        """Update existing document."""
        self._open_db()

        filepath = args.filepath
        keywords_file = args.keywords

        if not keywords_file:
            kb_path = Path(filepath)
            keywords_file = kb_path.with_suffix(kb_path.suffix + ".keywords.json")

        try:
            kw_data = KeywordParser.parse_file(keywords_file)
        except Exception as e:
            print(f"Error parsing keywords file: {e}", file=sys.stderr)
            return 1

        # Update document
        title = kw_data.get("title")
        if not title:
            try:
                title = MarkdownParser.extract_title(filepath)
            except Exception:
                pass

        success = self.db.update_document(
            filepath=kw_data["filepath"],
            title=title,
            summary=kw_data.get("summary"),
        )

        if not success:
            print(f"Document not found: {kw_data['filepath']}", file=sys.stderr)
            return 1

        # Update keywords - replace all keywords with new set from keywords file
        categories = kw_data.get("categories", {})
        keywords_data = []
        for keyword in kw_data["keywords"]:
            # Find category for this keyword
            category = None
            for cat_name, cat_keywords in categories.items():
                if keyword.lower() in [k.lower() for k in cat_keywords]:
                    category = cat_name
                    break
            keywords_data.append((keyword, category))

        self.db.replace_document_keywords(kw_data["filepath"], keywords_data)

        print(f"Updated document: {kw_data['filepath']}")
        return 0

    def cmd_remove(self, args):
        """Remove document from index."""
        self._open_db()

        success = self.db.remove_document(args.filepath)

        if success:
            print(f"Removed document: {args.filepath}")
            return 0
        else:
            print(f"Document not found: {args.filepath}", file=sys.stderr)
            return 1

    def cmd_list_docs(self, args):
        """List all indexed documents."""
        self._open_db()

        docs = self.db.list_documents()

        # Convert UTC timestamps to local time
        for doc in docs:
            if 'created_at' in doc:
                doc['created_at'] = self.utc_to_local(doc['created_at'])
            if 'updated_at' in doc:
                doc['updated_at'] = self.utc_to_local(doc['updated_at'])

        if args.format == "json":
            print(json.dumps(docs, indent=2))
        else:
            # Table format
            if not docs:
                print("No documents found.")
            else:
                print(f"{'Filepath':<50} {'Title':<40}")
                print("-" * 90)
                for doc in docs:
                    title = doc.get("title") or ""
                    print(f"{doc['filepath']:<50} {title:<40}")

        return 0

    def cmd_show(self, args):
        """Show document details."""
        self._open_db()

        search = SearchEngine(self.db)
        details = search.format_document_details(args.filepath)

        if not details:
            print(f"Document not found: {args.filepath}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(details, indent=2))
        else:
            # Convert UTC timestamps to local time
            created_local = self.utc_to_local(details['created_at'])
            updated_local = self.utc_to_local(details['updated_at'])

            print(f"Filepath: {details['filepath']}")
            print(f"Title: {details.get('title') or 'N/A'}")
            print(f"Summary: {details.get('summary') or 'N/A'}")
            print(f"Created: {created_local}")
            print(f"Updated: {updated_local}")
            print("\nKeywords:")
            for kw in details["keywords"]:
                category = f" ({kw['category']})" if kw.get("category") else ""
                print(f"  - {kw['keyword']}{category}")

        return 0

    # ==================== Keyword Commands ====================

    def cmd_keywords(self, args):
        """List keywords for a document."""
        self._open_db()

        keywords = self.db.get_document_keywords(args.filepath)

        if args.format == "json":
            print(json.dumps(keywords, indent=2))
        else:
            if not keywords:
                print("No keywords found.")
            else:
                print(f"Keywords for {args.filepath}:")
                for kw in keywords:
                    category = f" ({kw['category']})" if kw.get("category") else ""
                    print(f"  - {kw['keyword']}{category}")

        return 0

    def cmd_docs(self, args):
        """List documents containing keyword."""
        self._open_db()

        docs = self.db.get_documents_by_keyword(args.keyword)

        if args.format == "json":
            print(json.dumps(docs, indent=2))
        else:
            if not docs:
                print(f"No documents found with keyword: {args.keyword}")
            else:
                print(f"Documents with keyword '{args.keyword}':")
                for doc in docs:
                    title = f" - {doc['title']}" if doc.get("title") else ""
                    print(f"  {doc['filepath']}{title}")

        return 0

    def cmd_stats(self, args):
        """Get keyword statistics."""
        self._open_db()

        stats = self.db.get_keyword_stats(args.keyword)

        if not stats:
            print(f"Keyword not found: {args.keyword}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(json.dumps(stats, indent=2))
        else:
            print(f"Keyword: {stats['keyword']}")
            print(f"Documents: {stats['document_count']}")
            print(f"Related keywords: {stats['related_keywords_count']}")

        return 0

    # ==================== Similarity Commands ====================

    def cmd_similar(self, args):
        """Get similar keywords."""
        self._open_db()

        search = SearchEngine(self.db)
        similarities = self.db.get_similar_keywords(args.keyword, args.type)

        # Apply context filtering if user context is provided
        if args.user_context:
            try:
                from kb_indexer.context_matcher import ContextMatcher
                matcher = ContextMatcher(
                    backend=args.llm_backend,
                    model=args.llm_model
                )

                filtered_similarities = []
                for sim in similarities:
                    matches, score = matcher.matches(
                        args.keyword,
                        sim['related_keyword'],
                        sim['similarity_type'],
                        sim['context'],
                        args.user_context
                    )

                    if matches and score >= args.context_threshold:
                        # Add context match score to similarity
                        sim['context_match_score'] = score
                        filtered_similarities.append(sim)

                similarities = filtered_similarities
            except ImportError:
                print("Warning: google-genai not installed. Install with: pip install google-genai python-dotenv",
                      file=sys.stderr)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Context filtering failed: {e}", file=sys.stderr)

        result = search.format_similar_keywords(args.keyword, similarities)

        # Add user_context to result if provided
        if args.user_context:
            result['user_context'] = args.user_context

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            if not similarities:
                print(f"No similar keywords found for: {args.keyword}")
            else:
                context_note = f" (filtered by context: '{args.user_context}')" if args.user_context else ""
                print(f"Similar keywords for '{args.keyword}'{context_note}:")
                for sim in result["similar_keywords"]:
                    direction = " (one-way)" if sim["directional"] else ""
                    context_match = f" [match: {sim['context_match_score']:.2f}]" if 'context_match_score' in sim else ""
                    print(f"  [{sim['similarity_type']}] {sim['keyword']} (score: {sim['score']:.2f}){direction}{context_match}")
                    print(f"    → {sim['context']}")

        return 0

    def cmd_relate(self, args):
        """Add similarity relationship."""
        self._open_db()

        try:
            self.db.add_similarity(
                keyword1=args.keyword1,
                keyword2=args.keyword2,
                similarity_type=args.type,
                context=args.context,
                score=args.score,
                directional=args.directional,
            )
            print(f"Added similarity: {args.keyword1} ↔ {args.keyword2}")
            return 0
        except Exception as e:
            print(f"Error adding similarity: {e}", file=sys.stderr)
            return 1

    def cmd_unrelate(self, args):
        """Remove similarity relationship."""
        self._open_db()

        similarity_type = getattr(args, 'type', None)
        success = self.db.remove_similarity(args.keyword1, args.keyword2, similarity_type)

        if success:
            if similarity_type:
                print(f"Removed similarity ({similarity_type}): {args.keyword1} ↔ {args.keyword2}")
            else:
                print(f"Removed all similarities: {args.keyword1} ↔ {args.keyword2}")
            return 0
        else:
            if similarity_type:
                print(f"Similarity ({similarity_type}) not found: {args.keyword1} ↔ {args.keyword2}", file=sys.stderr)
            else:
                print(f"Similarity not found: {args.keyword1} ↔ {args.keyword2}", file=sys.stderr)
            return 1

    def cmd_import_similarities(self, args):
        """Import similarities from JSON file."""
        self._open_db()

        try:
            similarities = SimilarityParser.parse_file(args.file)

            count = 0
            for sim in similarities:
                self.db.add_similarity(
                    keyword1=sim["keyword1"],
                    keyword2=sim["keyword2"],
                    similarity_type=sim["type"],
                    context=sim["context"],
                    score=sim.get("score", 0.5),
                    directional=sim.get("directional", False),
                )
                count += 1

            print(f"Imported {count} similarity relationships")
            return 0

        except Exception as e:
            print(f"Error importing similarities: {e}", file=sys.stderr)
            return 1

    # ==================== Search Commands ====================

    def cmd_search(self, args):
        """Search for documents by keywords."""
        self._open_db()

        search = SearchEngine(self.db)

        # Determine search mode
        if args.and_mode:
            results = search.search_by_keywords_and(args.keywords)
            mode = "and"
        elif len(args.keywords) == 1:
            results = search.search_by_keyword(args.keywords[0])
            mode = "exact"
        else:
            results = search.search_by_keywords_or(args.keywords)
            mode = "or"

        formatted = search.format_search_results(results, args.keywords, mode)

        if args.format == "json":
            print(json.dumps(formatted, indent=2))
        else:
            if not results:
                print(f"No documents found matching keywords: {', '.join(args.keywords)}")
            else:
                print(f"Found {len(results)} document(s):\n")
                for doc in formatted["results"]:
                    print(f"📄 {doc['filepath']}")
                    if doc.get("title"):
                        print(f"   Title: {doc['title']}")
                    if doc.get("summary"):
                        summary = doc["summary"][:100] + "..." if len(doc["summary"]) > 100 else doc["summary"]
                        print(f"   Summary: {summary}")

                    # Show user keywords and matched keywords
                    user_kws = doc.get('user_keywords', [])
                    matched_kws = doc.get('matched_keywords', [])

                    if user_kws:
                        # Show which user keyword found this document
                        print(f"   Found by: {', '.join(user_kws)}")
                    if matched_kws:
                        print(f"   Matched keywords: {', '.join(matched_kws)}")
                    print()

        return 0

    def cmd_query(self, args):
        """Query documents with question, keywords, and context."""
        self._open_db()

        # Determine knowledge base path
        kb_path = args.kb_path or "./knowledge-base"

        try:
            # Initialize query engine
            query_engine = QueryEngine(
                db=self.db,
                knowledge_base_path=kb_path,
                backend=args.llm_backend,
                model=args.llm_model
            )

            # Execute query
            result = query_engine.query(
                question=args.question,
                keywords=args.keywords,
                context=args.context,
                threshold=args.threshold,
                expand_depth=args.expand_depth,
                enable_grep_fallback=not args.no_grep,
                enable_learning=not args.no_learn,
                auto_apply=not args.suggest_only
            )

            if args.format == "json":
                print(json.dumps(result, indent=2))
            else:
                # Human-readable output
                print(f"Query: {args.question}")
                print(f"Context: {args.context}")
                print(f"Keywords: {', '.join(args.keywords)}")
                if result["query"].get("expanded_keywords") and args.expand_depth > 0:
                    expanded = result["query"]["expanded_keywords"]
                    added = set(expanded) - set(args.keywords)
                    if added:
                        print(f"Expanded: +{len(added)} similar keywords ({', '.join(sorted(added)[:5])}{'...' if len(added) > 5 else ''})")
                print(f"Threshold: {args.threshold}\n")

                if result["count"] == 0:
                    print("No relevant documents found.")
                else:
                    print(f"Found {result['count']} relevant document(s):")
                    print(f"  - Keyword search: {result['keyword_search_count']}")
                    print(f"  - Grep search: {result['grep_search_count']}\n")

                    for doc in result["results"]:
                        source_icon = "🔍" if doc["source"] == "keyword_search" else "📝"

                        # Show indexing status
                        if doc.get("auto_indexed"):
                            indexed_note = " [AUTO-INDEXED ✓]"
                        elif doc.get("reindexed"):
                            indexed_note = " [REINDEXED ↻]"
                        elif not doc.get("indexed", True):
                            indexed_note = " [NOT INDEXED]"
                        else:
                            indexed_note = ""

                        print(f"{source_icon} {doc['filepath']}{indexed_note}")
                        if doc.get("title"):
                            print(f"   Title: {doc['title']}")
                        if doc.get("summary"):
                            summary = doc["summary"][:100] + "..." if len(doc["summary"]) > 100 else doc["summary"]
                            print(f"   Summary: {summary}")
                        print(f"   Relevance: {doc['relevance_score']:.2f}")
                        print(f"   Reasoning: {doc['reasoning']}")

                        # Show keyword expansion information
                        if doc.get("keyword_expansions"):
                            # Document was found via expanded keywords
                            expansions_str = []
                            for exp in doc["keyword_expansions"]:
                                expansions_str.append(f"{exp['original']} → {exp['expanded']}")
                            print(f"   Found by: {', '.join(expansions_str)}")
                        elif doc.get("user_keywords"):
                            # Document was found via original user keywords
                            print(f"   Found by: {', '.join(doc['user_keywords'])}")

                        if doc.get("matched_keywords"):
                            print(f"   Doc keywords: {', '.join(doc['matched_keywords'])}")
                        print()

                # Show learning suggestions and application status
                if result.get("suggestions"):
                    suggestions = result["suggestions"]
                    kw_suggestions = suggestions.get("keyword_suggestions", [])
                    sim_suggestions = suggestions.get("similarity_suggestions", [])

                    if kw_suggestions or sim_suggestions:
                        applied = result.get("applied")
                        if applied:
                            print("=== Learning: Suggestions Applied ===\n")
                            print(f"✓ Keywords added: {applied['keywords_added']}")
                            print(f"✓ Similarities added: {applied['similarities_added']}")
                            if applied.get('errors'):
                                print(f"✗ Errors: {len(applied['errors'])}")
                                for error in applied['errors'][:3]:  # Show first 3 errors
                                    print(f"  - {error}")
                            print()
                        else:
                            print("=== Learning Suggestions (Preview) ===\n")

                        if kw_suggestions:
                            print("Keyword Suggestions:")
                            for sugg in kw_suggestions:
                                status = "✓ APPLIED" if applied else ""
                                print(f"  📄 {sugg['filepath']} {status}")
                                print(f"     Add keywords: {', '.join(sugg['keywords'])}")
                                print(f"     Reasoning: {sugg['reasoning']}\n")

                        if sim_suggestions:
                            print("Similarity Suggestions:")
                            for sugg in sim_suggestions:
                                status = "✓ APPLIED" if applied else ""
                                print(f"  🔗 {sugg['keyword1']} ↔ {sugg['keyword2']} {status}")
                                print(f"     Type: {sugg['type']}")
                                print(f"     Context: {sugg['context']}")
                                print(f"     Score: {sugg['score']:.2f}")
                                print(f"     Reasoning: {sugg['reasoning']}\n")

            return 0

        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Install required packages: pip install ollama google-genai", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error executing query: {e}", file=sys.stderr)
            return 1

    # ==================== Database Commands ====================

    def cmd_init(self, args):
        """Initialize new database."""
        db_path = args.db or self.db_path

        if Path(db_path).exists():
            print(f"Database already exists: {db_path}", file=sys.stderr)
            return 1

        # Find schema.sql
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            print(f"Schema file not found: {schema_path}", file=sys.stderr)
            return 1

        # Initialize database
        db = Database(db_path)
        db.init_schema(schema_path)
        db.close()

        print(f"Initialized database: {db_path}")
        return 0

    def cmd_db_stats(self, args):
        """Get database statistics."""
        self._open_db()

        stats = self.db.get_stats()

        print(f"Database statistics:")
        print(f"  Documents: {stats['documents']}")
        print(f"  Keywords: {stats['keywords']}")
        print(f"  Similarities: {stats['similarities']}")

        return 0

    # ==================== Main ====================

    def run(self):
        """Run CLI."""
        parser = argparse.ArgumentParser(
            description="Knowledge Base Indexer - Index and search documents with keywords"
        )
        parser.add_argument("--db", help="Database path (default: kb_index.db)")

        subparsers = parser.add_subparsers(dest="command", help="Command")

        # Document commands
        p_add = subparsers.add_parser("add", help="Add document to index",
                                      description="Add a document to the index with keywords from JSON file")
        p_add.add_argument("filepath", help="Path to markdown document (e.g., knowledge-base/doc.md or ./knowledge-base/doc.md)")
        p_add.add_argument("--keywords", help="Path to keywords JSON file (default: <filepath>.keywords.json)")

        p_update = subparsers.add_parser("update", help="Update existing document",
                                         description="Update an existing document's keywords and metadata")
        p_update.add_argument("filepath", help="Path to markdown document (e.g., knowledge-base/doc.md or ./knowledge-base/doc.md)")
        p_update.add_argument("--keywords", help="Path to keywords JSON file (default: <filepath>.keywords.json)")

        p_remove = subparsers.add_parser("remove", help="Remove document from index",
                                         description="Remove a document and all its keywords from the index")
        p_remove.add_argument("filepath", help="Document filepath as stored in database (e.g., doc.md)")

        p_list_docs = subparsers.add_parser("list-docs", help="List all indexed documents",
                                            description="List all documents in the index with titles and timestamps")
        p_list_docs.add_argument("--format", choices=["json", "table"], default="table",
                                 help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        p_show = subparsers.add_parser("show", help="Show document details",
                                       description="Show detailed information about a document including all keywords")
        p_show.add_argument("filepath", help="Document filepath as stored in database (e.g., doc.md)")
        p_show.add_argument("--format", choices=["json", "table"], default="table",
                           help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        # Keyword commands
        p_keywords = subparsers.add_parser("keywords", help="List keywords for document",
                                           description="List all keywords associated with a document")
        p_keywords.add_argument("filepath", help="Document filepath as stored in database (e.g., doc.md)")
        p_keywords.add_argument("--format", choices=["json", "table"], default="table",
                               help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        p_docs = subparsers.add_parser("docs", help="List documents containing keyword",
                                       description="Find all documents that contain a specific keyword")
        p_docs.add_argument("keyword", help="Keyword to search for (e.g., 'reinforcement learning')")
        p_docs.add_argument("--format", choices=["json", "table"], default="table",
                           help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        p_stats = subparsers.add_parser("stats", help="Get keyword statistics",
                                        description="Get statistics about a keyword (document count, related keywords)")
        p_stats.add_argument("keyword", help="Keyword to get statistics for (e.g., 'AGI')")
        p_stats.add_argument("--format", choices=["json", "table"], default="table",
                            help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        # Similarity commands
        p_similar = subparsers.add_parser("similar", help="Get similar keywords",
                                          description="Find keywords similar to a given keyword with optional context filtering",
                                          epilog="""
Context threshold guidelines (when using --user-context):
  0.9-1.0 = Very strict filtering (only highly relevant matches)
  0.7-0.8 = Balanced filtering (recommended, filters out unrelated)
  0.5-0.6 = Relaxed filtering (includes loosely related terms)
  0.3-0.4 = Minimal filtering (keeps most results)

Examples:
  ./kbindex.py similar "RL" --user-context "game AI" --context-threshold 0.7
  ./kbindex.py similar "Python" --user-context "package management" --type related
                                          """,
                                          formatter_class=argparse.RawDescriptionHelpFormatter)
        p_similar.add_argument("keyword", help="Keyword to find similarities for (e.g., 'RL')")
        p_similar.add_argument("--type", help="Filter by similarity type (e.g., 'abbreviation', 'synonym', 'related')")
        p_similar.add_argument("--user-context", help="User's context for AI-powered filtering (e.g., 'game AI and competitions')")
        p_similar.add_argument("--context-threshold", type=float, default=0.7,
                              help="Context match threshold: 0.9-1.0=very strict, 0.7-0.8=balanced (default), 0.5-0.6=relaxed")
        p_similar.add_argument("--llm-backend", choices=["gemini", "ollama"], default="ollama",
                              help="LLM backend for context matching: 'ollama' (local, free, default) or 'gemini' (cloud, requires API key)")
        p_similar.add_argument("--llm-model", help="LLM model name (e.g., 'llama3.2:3b' for ollama, default: auto-select)")
        p_similar.add_argument("--format", choices=["json", "table"], default="table",
                              help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        p_relate = subparsers.add_parser("relate", help="Add similarity relationship",
                                         description="Add a similarity relationship between two keywords",
                                         epilog="""
Score guidelines:
  1.0 = Perfect match (abbreviations, exact synonyms)
  0.8-0.9 = Very strong relationship (close synonyms, common alternates)
  0.6-0.7 = Strong relationship (related concepts, domain-specific terms)
  0.4-0.5 = Moderate relationship (loosely related, broader associations)
  0.2-0.3 = Weak relationship (tangentially related)

Common similarity types:
  - abbreviation: Shortened form (RL -> reinforcement learning, AI -> artificial intelligence)
  - synonym: Same meaning, different wording (neural network <-> artificial neural network)
  - related: Closely related concepts (neural networks -> deep learning)
  - broader: More general term (artificial intelligence -> machine learning)
  - narrower: More specific term (CNN -> neural networks)
  - alternative: Alternative naming (car -> automobile, JS -> JavaScript)

Context examples (describe the domain/background):
  - "machine learning and AI research"
  - "game AI and board game competitions"
  - "natural language processing applications"
  - "reinforcement learning agent design"
                                         """,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        p_relate.add_argument("keyword1", help="First keyword (e.g., 'RL')")
        p_relate.add_argument("keyword2", help="Second keyword (e.g., 'reinforcement learning')")
        p_relate.add_argument("--type", required=True,
                             help="Similarity type: abbreviation, synonym, related, broader, narrower, alternative")
        p_relate.add_argument("--context", required=True, help="Background context that explains the meaning of keywords and where they're used (e.g., 'machine learning and AI research', 'game AI competitions')")
        p_relate.add_argument("--score", type=float, default=0.5,
                             help="Similarity strength: 1.0=perfect, 0.8-0.9=very strong, 0.6-0.7=strong, 0.4-0.5=moderate (default: 0.5)")
        p_relate.add_argument("--directional", action="store_true", help="One-way relationship (keyword1 -> keyword2 only)")

        p_unrelate = subparsers.add_parser("unrelate", help="Remove similarity relationship",
                                           description="Remove a similarity relationship between two keywords")
        p_unrelate.add_argument("keyword1", help="First keyword (e.g., 'RL')")
        p_unrelate.add_argument("keyword2", help="Second keyword (e.g., 'reinforcement learning')")
        p_unrelate.add_argument("--type", help="Similarity type to remove (if not specified, removes all types)")

        p_import_sim = subparsers.add_parser("import-similarities", help="Import similarities from JSON",
                                             description="Import multiple similarity relationships from a JSON file")
        p_import_sim.add_argument("file", help="Path to similarities JSON file (e.g., examples/similarities.json)")

        # Search commands
        p_search = subparsers.add_parser("search", help="Search for documents by keywords",
                                         description="Search for documents matching one or more keywords")
        p_search.add_argument("keywords", nargs="+", help="One or more keywords to search for (e.g., 'LLM' 'reinforcement learning')")
        p_search.add_argument("--or", dest="or_mode", action="store_true", help="OR mode: match any keyword (default for multiple keywords)")
        p_search.add_argument("--and", dest="and_mode", action="store_true", help="AND mode: match all keywords")
        p_search.add_argument("--format", choices=["json", "table"], default="table",
                             help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        p_query = subparsers.add_parser("query", help="Query documents with question and context",
                                        description="Intelligent document query with LLM-based filtering, grep fallback, and learning",
                                        epilog="""
This command combines keyword search, LLM-based relevance filtering, and automatic learning.

Workflow:
1. EXPAND keywords using similarity relationships (1 level by default)
2. Search documents by expanded keywords
3. Score each document's relevance to your question using LLM
4. If no results, fall back to grep search across all files
5. AUTO-INDEX any unindexed documents found via grep
6. Generate and AUTO-APPLY learning suggestions to improve the index

Keyword Expansion (default: 1 level):
  - Automatically finds similar keywords (e.g., "RL" → "reinforcement learning")
  - Use --expand-depth 0 to disable expansion
  - Use --expand-depth 2 for two levels of expansion (finds similar keywords of similar keywords)

Learning behavior (default: auto-apply):
  - By default, suggestions are automatically applied to the database
  - Use --suggest-only to preview suggestions without applying
  - Use --no-learn to disable learning entirely

Threshold guidelines:
  0.9-1.0 = Very strict (only documents that directly answer the question)
  0.7-0.8 = Balanced (recommended, filters out unrelated documents)
  0.5-0.6 = Relaxed (includes tangentially related documents)
  0.3-0.4 = Very relaxed (includes loosely related documents)

Examples:
  # Basic query with keywords
  ./kbindex.py query "How does AlphaGo use RL?" \\
    --keywords "reinforcement learning" "game AI" \\
    --context "board games and competitions"

  # With custom threshold and grep fallback
  ./kbindex.py query "What is Q-learning?" \\
    --keywords "RL" "learning" \\
    --context "reinforcement learning algorithms" \\
    --threshold 0.8

  # Preview suggestions without applying (review mode)
  ./kbindex.py query "Explain neural networks" \\
    --keywords "neural networks" "deep learning" \\
    --context "machine learning" \\
    --suggest-only

  # Disable keyword expansion (exact match only)
  ./kbindex.py query "What is RL?" \\
    --keywords "RL" \\
    --context "machine learning" \\
    --expand-depth 0

  # Deep expansion (2 levels)
  ./kbindex.py query "What is RL?" \\
    --keywords "RL" \\
    --context "machine learning" \\
    --expand-depth 2

  # Disable learning entirely
  ./kbindex.py query "What are GANs?" \\
    --keywords "GAN" \\
    --context "generative models" \\
    --no-learn

  # Use alternative LLM backend (default is claude)
  ./kbindex.py query "What is DQN?" \\
    --keywords "deep learning" "Q-learning" \\
    --context "reinforcement learning" \\
    --llm-backend ollama

  # JSON output for programmatic use
  ./kbindex.py query "What are transformers?" \\
    --keywords "transformer" "attention" \\
    --context "NLP" \\
    --format json
                                        """,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
        p_query.add_argument("question", help="Your question about the topic")
        p_query.add_argument("--keywords", nargs="+", required=True,
                            help="Keywords to search for (e.g., 'reinforcement learning' 'AlphaGo')")
        p_query.add_argument("--context", required=True,
                            help="Context/domain for filtering (e.g., 'game AI and competitions')")
        p_query.add_argument("--threshold", type=float, default=0.7,
                            help="Minimum relevance score 0.0-1.0 (default: 0.7)")
        p_query.add_argument("--expand-depth", type=int, default=1,
                            help="Keyword expansion depth using similarities (0=no expansion, 1=one level, etc., default: 1)")
        p_query.add_argument("--no-grep", action="store_true",
                            help="Disable grep fallback search")
        p_query.add_argument("--no-learn", action="store_true",
                            help="Disable learning suggestions entirely")
        p_query.add_argument("--suggest-only", action="store_true",
                            help="Generate suggestions but don't auto-apply (preview mode)")
        p_query.add_argument("--kb-path", help="Path to knowledge base directory (default: ./knowledge-base)")
        p_query.add_argument("--llm-backend", choices=["claude", "gemini", "ollama"], default="ollama",
                            help="LLM backend: 'ollama' (local, default), 'claude' (Claude Code CLI), or 'gemini' (cloud)")
        p_query.add_argument("--llm-model", help="LLM model name (default: auto-select based on backend)")
        p_query.add_argument("--format", choices=["json", "table"], default="table",
                            help="Output format: 'json' for structured data, 'table' for human-readable (default: table)")

        # Database commands
        p_init = subparsers.add_parser("init", help="Initialize new database",
                                       description="Create a new database with schema")

        p_db_stats = subparsers.add_parser("db-stats", help="Database statistics",
                                           description="Show statistics about the database (document count, keyword count, etc.)")

        args = parser.parse_args()

        # Override db_path if specified
        if args.db:
            self.db_path = args.db

        # Dispatch command
        if not args.command:
            parser.print_help()
            return 2

        command_map = {
            "add": self.cmd_add,
            "update": self.cmd_update,
            "remove": self.cmd_remove,
            "list-docs": self.cmd_list_docs,
            "show": self.cmd_show,
            "keywords": self.cmd_keywords,
            "docs": self.cmd_docs,
            "stats": self.cmd_stats,
            "similar": self.cmd_similar,
            "relate": self.cmd_relate,
            "unrelate": self.cmd_unrelate,
            "import-similarities": self.cmd_import_similarities,
            "search": self.cmd_search,
            "query": self.cmd_query,
            "init": self.cmd_init,
            "db-stats": self.cmd_db_stats,
        }

        try:
            return command_map[args.command](args)
        finally:
            self._close_db()


def main():
    """Main entry point."""
    cli = CLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
