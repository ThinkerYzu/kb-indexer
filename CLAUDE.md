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
│   └── context_matcher.py  # LLM-based context matching
├── tests/                  # Test suite (59 tests)
├── scripts/                # Helper scripts
└── examples/               # Example data files
```
