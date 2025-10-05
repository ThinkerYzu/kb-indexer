#!/usr/bin/env python3
"""
Knowledge Base Indexer CLI

Command-line interface for indexing and searching documents in knowledge-base/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from kb_indexer import Database, KeywordParser, MarkdownParser, SearchEngine
from kb_indexer.parser import SimilarityParser


class CLI:
    """Command-line interface for kbindex."""

    def __init__(self, db_path: str = "kb_index.db"):
        """Initialize CLI.

        Args:
            db_path: Path to SQLite database
        """
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

        success = self.db.remove_similarity(args.keyword1, args.keyword2)

        if success:
            print(f"Removed similarity: {args.keyword1} ↔ {args.keyword2}")
            return 0
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
                    print(f"   Matched: {', '.join(doc['matched_keywords'])}")
                    print()

        return 0

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
        p_add = subparsers.add_parser("add", help="Add document to index")
        p_add.add_argument("filepath", help="Path to document")
        p_add.add_argument("--keywords", help="Path to keywords.json file")

        p_update = subparsers.add_parser("update", help="Update existing document")
        p_update.add_argument("filepath", help="Path to document")
        p_update.add_argument("--keywords", help="Path to keywords.json file")

        p_remove = subparsers.add_parser("remove", help="Remove document from index")
        p_remove.add_argument("filepath", help="Path to document")

        p_list_docs = subparsers.add_parser("list-docs", help="List all indexed documents")
        p_list_docs.add_argument("--format", choices=["json", "table"], default="table")

        p_show = subparsers.add_parser("show", help="Show document details")
        p_show.add_argument("filepath", help="Path to document")
        p_show.add_argument("--format", choices=["json", "table"], default="table")

        # Keyword commands
        p_keywords = subparsers.add_parser("keywords", help="List keywords for document")
        p_keywords.add_argument("filepath", help="Path to document")
        p_keywords.add_argument("--format", choices=["json", "table"], default="table")

        p_docs = subparsers.add_parser("docs", help="List documents containing keyword")
        p_docs.add_argument("keyword", help="Keyword to search")
        p_docs.add_argument("--format", choices=["json", "table"], default="table")

        p_stats = subparsers.add_parser("stats", help="Get keyword statistics")
        p_stats.add_argument("keyword", help="Keyword")
        p_stats.add_argument("--format", choices=["json", "table"], default="table")

        # Similarity commands
        p_similar = subparsers.add_parser("similar", help="Get similar keywords")
        p_similar.add_argument("keyword", help="Keyword")
        p_similar.add_argument("--type", help="Filter by similarity type")
        p_similar.add_argument("--user-context", help="User's context for AI-powered filtering")
        p_similar.add_argument("--context-threshold", type=float, default=0.7,
                              help="Minimum context match score (0.0-1.0, default: 0.7)")
        p_similar.add_argument("--llm-backend", choices=["gemini", "ollama"], default="ollama",
                              help="LLM backend for context matching (default: ollama)")
        p_similar.add_argument("--llm-model", help="LLM model name (default: auto-select)")
        p_similar.add_argument("--format", choices=["json", "table"], default="table")

        p_relate = subparsers.add_parser("relate", help="Add similarity relationship")
        p_relate.add_argument("keyword1", help="First keyword")
        p_relate.add_argument("keyword2", help="Second keyword")
        p_relate.add_argument("--type", required=True, help="Similarity type")
        p_relate.add_argument("--context", required=True, help="Context explanation")
        p_relate.add_argument("--score", type=float, default=0.5, help="Score (0-1)")
        p_relate.add_argument("--directional", action="store_true", help="One-way relationship")

        p_unrelate = subparsers.add_parser("unrelate", help="Remove similarity relationship")
        p_unrelate.add_argument("keyword1", help="First keyword")
        p_unrelate.add_argument("keyword2", help="Second keyword")

        p_import_sim = subparsers.add_parser("import-similarities", help="Import similarities from JSON")
        p_import_sim.add_argument("file", help="Path to similarities.json")

        # Search commands
        p_search = subparsers.add_parser("search", help="Search for documents by keywords")
        p_search.add_argument("keywords", nargs="+", help="Keywords to search")
        p_search.add_argument("--or", dest="or_mode", action="store_true", help="OR mode (default for multiple keywords)")
        p_search.add_argument("--and", dest="and_mode", action="store_true", help="AND mode")
        p_search.add_argument("--format", choices=["json", "table"], default="table")

        # Database commands
        p_init = subparsers.add_parser("init", help="Initialize new database")
        p_init.add_argument("--db", help="Database path")

        p_db_stats = subparsers.add_parser("db-stats", help="Database statistics")

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
