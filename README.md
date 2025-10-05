# Knowledge Base Indexer (kbindex)

A pure data tool for indexing and searching documents with keywords and semantic relationships. Designed to provide structured data for AI agents to make intelligent search decisions.

## Overview

**kbindex** stores documents, keywords, and their relationships in a SQLite database. It provides query interfaces that return structured data (JSON) for AI agents to interpret and use in search expansion and decision-making.

**Core Principle:** kbindex is a data tool, not an AI. It stores and retrieves information; AI agents make all intelligent decisions.

## Features

- **Document Indexing** - Store markdown documents with titles, summaries, and keywords
- **Keyword Management** - Normalized keyword storage with optional categorization
- **Semantic Relationships** - Define similarities between keywords with context and scoring
- **AI-Friendly Output** - JSON format with rich metadata for AI decision-making
- **Simple CLI** - Easy-to-use command-line interface
- **No Dependencies** - Pure Python with stdlib only

## Quick Start

```bash
# Initialize database
./kbindex.py init

# Add a document
./kbindex.py add examples/ai-llm-vs-reinforcement-learning.md \
  --keywords examples/sample.keywords.json

# Import similarity relationships
./kbindex.py import-similarities examples/similarities.json

# Search for documents
./kbindex.py search "reinforcement learning" --format json

# Find similar keywords
./kbindex.py similar "reinforcement learning" --format json
```

## Installation

No installation required. Just Python 3.8+:

```bash
cd kb-indexer
chmod +x kbindex.py
./kbindex.py --help
```

## Documentation

- [DESIGN.md](DESIGN.md) - Comprehensive design document
- [schema.sql](schema.sql) - Database schema
- [examples/](examples/) - Example keyword and similarity files

## Usage Examples

### Indexing Documents

```bash
# Sync entire knowledge-base directory (recommended)
./scripts/sync_kb.sh

# Add single document
./kbindex.py add ../knowledge-base/my-doc.md

# Update existing document
./kbindex.py update ../knowledge-base/my-doc.md

# Remove document from index
./kbindex.py remove ../knowledge-base/my-doc.md

# List all indexed documents (timestamps shown in local time)
./kbindex.py list-docs

# Show document details (timestamps shown in local time)
./kbindex.py show ../knowledge-base/my-doc.md
```

### Searching

```bash
# Search by single keyword
./kbindex.py search "AGI"

# Multiple keywords (OR - default)
./kbindex.py search "LLM" "reinforcement learning" --or

# Multiple keywords (AND - all must match)
./kbindex.py search "LLM" "AGI" --and

# Get results as JSON
./kbindex.py search "AGI" --format json
```

### Working with Similarities

```bash
# Find similar keywords
./kbindex.py similar "RL" --format json

# Filter by similarity type
./kbindex.py similar "reinforcement learning" --type abbreviation

# Context-aware similarity search (AI-powered filtering)
# Default: Uses local Ollama (free, no API key needed)
./kbindex.py similar "RL" --user-context "game AI and competitions" --format json
./kbindex.py similar "Python" --user-context "package management" --context-threshold 0.8

# Choose LLM backend
./kbindex.py similar "RL" --user-context "robotics" --llm-backend ollama  # Local (default)
./kbindex.py similar "RL" --user-context "robotics" --llm-backend gemini  # Cloud (requires API key)

# Use custom model
./kbindex.py similar "RL" --user-context "game AI" --llm-backend ollama --llm-model llama3.2:3b

# Add a similarity relationship
./kbindex.py relate "RL" "reinforcement learning" \
  --type abbreviation \
  --context "RL is abbreviation for reinforcement learning" \
  --score 1.0

# Remove similarity relationship
./kbindex.py unrelate "RL" "reinforcement learning"

# Import similarities from file
./kbindex.py import-similarities examples/similarities.json
```

### Keywords

```bash
# List keywords for a document
./kbindex.py keywords ../knowledge-base/my-doc.md

# Find documents containing a keyword
./kbindex.py docs "reinforcement learning"

# Get keyword statistics
./kbindex.py stats "reinforcement learning"
```

### Database Management

```bash
# Get database statistics
./kbindex.py db-stats

# Initialize new database at custom path
./kbindex.py init --db /path/to/custom.db
```

## File Formats

### Keywords JSON

Create `<filename>.keywords.json` alongside your markdown:

```json
{
  "filepath": "my-doc.md",
  "title": "Document Title",
  "summary": "Brief description of the document",
  "keywords": [
    "keyword1",
    "keyword2",
    "keyword3"
  ],
  "categories": {
    "primary": ["keyword1"],
    "concepts": ["keyword2", "keyword3"]
  }
}
```

### Similarities JSON

Define keyword relationships in `similarities.json`:

```json
{
  "similarities": [
    {
      "keyword1": "RL",
      "keyword2": "reinforcement learning",
      "type": "abbreviation",
      "context": "RL is abbreviation for reinforcement learning",
      "score": 1.0,
      "directional": false
    }
  ]
}
```

## For AI Agents

kbindex is designed to be used by AI agents. Example workflow:

### Basic Workflow

1. **User asks:** "Find documents about reinforcement learning"
2. **AI queries:** `kbindex similar "reinforcement learning" --format json`
3. **AI receives:** List of similar keywords with types, context, scores
4. **AI decides:** Which keywords to include based on context and user intent
5. **AI searches:** `kbindex search "reinforcement learning" "RL" --or --format json`
6. **AI presents:** Results to user

### Context-Aware Workflow

1. **User asks:** "Find documents about RL in game AI"
2. **AI queries:** `kbindex similar "RL" --user-context "game AI and competitions" --format json`
3. **AI receives:** Filtered list with context match scores
   - Uses local Ollama by default (free, no API key)
   - Can use cloud Gemini with `--llm-backend gemini` (requires API key)
4. **AI searches:** With contextually relevant keywords only (e.g., "AlphaGo", not general ML terms)
5. **AI presents:** More precise, domain-specific results

**LLM Backend Options:**
- **Ollama** (default): Local, free, no API key, privacy-preserving
- **Gemini**: Cloud, requires `GEMINI_API_KEY` environment variable

All output includes rich context for AI decision-making (similarity types, relevance scores, directional flags, context match scores).

## Project Structure

```
kb-indexer/
├── README.md                  # This file
├── DESIGN.md                  # Design documentation
├── IMPLEMENTATION.md          # Implementation details and completed features
├── schema.sql                 # Database schema
├── requirements.txt           # Python dependencies (optional: ollama, google-genai)
├── kbindex.py                 # Main CLI entry point
├── kb_indexer/
│   ├── __init__.py
│   ├── database.py           # SQLite database operations
│   ├── parser.py             # JSON/markdown parsing
│   ├── search.py             # Query and search operations
│   └── context_matcher.py    # LLM-based context matching (Ollama/Gemini)
├── scripts/
│   └── sync_kb.sh            # Sync knowledge-base directory to database
├── examples/
│   ├── sample.keywords.json
│   └── similarities.json
└── tests/
    ├── test_database.py
    ├── test_parser.py
    └── test_search.py
```

## Development

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_database.py -v

# Run with coverage
python3 -m pytest tests/ --cov=kb_indexer
```

### Test Results

All 40 unit tests pass:
- 16 database operation tests
- 12 parser tests (keywords, similarities, markdown)
- 12 search engine tests

### Code Organization

- **database.py** (600+ lines) - Complete SQLite CRUD operations
- **parser.py** (250+ lines) - JSON and markdown parsing with validation
- **search.py** (250+ lines) - Search, filtering, and output formatting
- **kbindex.py** (500+ lines) - CLI with all commands

## License

MIT License (or as per project license)

## See Also

- [DESIGN.md](DESIGN.md) - Detailed design decisions and architecture
- [../knowledge-base/](../knowledge-base/) - Document repository
