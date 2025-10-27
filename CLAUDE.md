# Claude Code Instructions for kb-indexer

This project is a standalone knowledge base indexer tool.

## Project Overview

kb-indexer is a pure data tool for indexing and searching documents with keywords and semantic relationships. It stores documents, keywords, and their relationships in a SQLite database and provides structured data (JSON) for AI agents to make intelligent search decisions.

## Key Design Principles

1. **Data Tool, Not AI**: kb-indexer stores and retrieves information; AI agents make all intelligent decisions
2. **Structured Output**: All commands provide JSON format for programmatic use
3. **Keyword-Based Search**: Uses exact keyword matching, NOT auto-expansion
4. **Semantic Relationships**: Define similarities between keywords with types, context, and scoring

## Development Guidelines

### Running Commands

Always check `--help` for detailed usage, especially for option values and guidelines:

```bash
./kbindex.py --help
./kbindex.py search --help
./kbindex.py relate --help
```

### Testing

Run the full test suite before committing changes:

```bash
python3 -m pytest tests/ -v
```

All 59 tests should pass:
- 16 database operation tests
- 12 parser tests
- 12 search engine tests
- 19 real data tests

### Code Quality

- Keep functions focused and single-purpose
- Add comprehensive docstrings
- Validate input parameters
- Use type hints where appropriate
- Follow existing code style

### Database Changes

When modifying schema.sql:
1. Update schema.sql first
2. Update database.py methods accordingly
3. Add/update tests in tests/test_database.py
4. Document changes in DESIGN.md or IMPLEMENTATION.md

### Similarity Types

The system supports multiple similarity types for the same keyword pair:
- `abbreviation`: Shortened form (e.g., "RL" → "reinforcement learning")
- `synonym`: Same meaning, different wording
- `related`: Closely related concepts
- `broader`: More general term
- `narrower`: More specific term
- `alternative`: Alternative naming

The UNIQUE constraint is on `(keyword_id_1, keyword_id_2, similarity_type)` to allow multiple types per keyword pair.

### Common Tasks

**Add new command:**
1. Add command method in CLI class (kbindex.py)
2. Add argument parser in run() method
3. Add command mapping in command_map dict
4. Add tests in appropriate test file
5. Update README.md with examples

**Fix bugs:**
1. Write a failing test first
2. Fix the code
3. Verify test passes
4. Check all tests still pass

**Refactor:**
1. Ensure all tests pass before starting
2. Make incremental changes
3. Run tests after each change
4. Update docstrings and comments

## Important Notes

- The `knowledge-base/` directory can be a symlink or actual directory (git-ignored)
- All timestamps are stored in UTC but displayed in local time
- Keywords are normalized (lowercase, trimmed) automatically
- The `context` field in similarities describes the domain/background, not the relationship
- Search uses exact keyword matching - use `similar` command to discover keywords first

## File Structure

```
kb-indexer/
├── kbindex.py              # Main CLI entry point
├── schema.sql              # Database schema
├── kb_indexer/
│   ├── database.py         # SQLite operations
│   ├── parser.py           # JSON/markdown parsing
│   ├── search.py           # Search engine
│   ├── query.py            # Intelligent query engine with LLM filtering
│   └── context_matcher.py  # LLM-based context matching
├── tests/                  # Test suite (70+ tests)
├── scripts/                # Helper scripts
└── examples/               # Example data files
```

## Query Command

The `query` command provides advanced question-based document retrieval:

**Features:**
- **Keyword expansion** using similarity relationships (default: 1 level)
- **Expansion tracking** - shows which user keyword expanded to which similar keyword
- LLM-powered relevance scoring of documents based on user's question
- Automatic grep fallback for searching unindexed files
- **AUTO-INDEXING** of unindexed documents (includes query keywords)
- **SMART LEARNING** - only triggers when keyword search fails but grep succeeds
  - Adds query keywords to found documents as similarities or direct keywords
  - Automatically applied by default
- Configurable relevance threshold (default: 0.7)
- Configurable expansion depth (default: 1)
- `--expand-depth 0` to disable expansion
- `--suggest-only` to preview suggestions without applying
- `--no-learn` to disable learning entirely

**Output Format:**
- Shows which user-provided keywords found each document
- For expanded keywords, displays the expansion chain (e.g., "RL → reinforcement learning")
- JSON output includes `expansion_map`, `user_keywords`, and `keyword_expansions` fields
- Human-readable output: "Found by: RL → reinforcement learning, AlphaGo"

**Usage:**
```bash
./kbindex.py query "How does AlphaGo work?" \
  --keywords "reinforcement learning" "AlphaGo" \
  --context "game AI" \
  --threshold 0.7
```

**Implementation Details:**
- QueryEngine (kb_indexer/query.py) orchestrates the workflow
- **Default LLM: Ollama** (also supports Claude Code CLI/Gemini)
- **Keyword expansion**: expand_keywords() method
  - Recursively expands keywords using database similarity relationships
  - Supports multi-level expansion (configurable depth)
  - Simple context filtering (30% word overlap threshold)
- Grep searches knowledge-base directory for files matching extracted terms
- **Auto-indexing**: Unindexed documents found via grep are automatically indexed
  - Uses LLM with question-based keyword generation (same approach as generate_keywords.py)
  - LLM first thinks about questions users might ask, then generates search keywords
  - **Includes query keywords** that led to finding the document
  - Documents are skipped if LLM-based indexing fails (no fallback)
  - auto_index_document(filepath, query_keywords) method handles the indexing
- **Auto-reindexing**: Documents modified since last indexing are automatically updated
  - Checks file modification time vs database timestamp
  - Uses LLM with question-based keyword generation to intelligently merge keywords
  - LLM decides which keywords to keep, add, or remove based on updated content
  - reindex_document_if_modified(filepath, query_keywords) method handles the reindexing
  - Marked with `[REINDEXED ↻]` in output
- **Smart Learning** (_generate_query_based_suggestions method):
  - **Only triggers** when keyword_search_count == 0 and grep_search_count > 0
  - For indexed documents: adds query keywords as **similarities** to existing keywords
  - For newly indexed documents: query keywords already added during auto-indexing
  - Reasoning: If grep found it but keywords didn't, the index needs those query keywords
- **Auto-apply by default**: New keywords and similarities are automatically added to database
  - Use `--suggest-only` to preview without applying
  - Use `--no-learn` to disable learning entirely
- Auto-indexed files are marked with `[AUTO-INDEXED ✓]` in output
- Reindexed files are marked with `[REINDEXED ↻]` in output
- Expanded keywords are shown in output (e.g., "Expanded: +3 similar keywords")
- apply_learning_suggestions() method handles database updates
- Duplicate prevention: checks existing keywords/similarities before adding

**LLM Backends:**
- `ollama` (default): Local Ollama - requires ollama installation and model download (free)
- `claude`: Claude Code CLI - requires `claude` command available
- `gemini`: Google Gemini API - requires GEMINI_API_KEY environment variable
