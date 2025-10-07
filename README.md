# Knowledge Base Indexer (kbindex)

Build a domain-specific semantic search layer using LLMs to automatically generate and maintain keywords and relationships over time.

## Overview

Manual keyword maintenance is impractical, and users rarely know the exact keywords needed to find documents. **kbindex** solves this by leveraging LLMs to build an ever-growing semantic layer:

- **LLMs generate keywords** automatically from documents
- **LLMs discover relationships** between keywords (abbreviations, synonyms, related concepts)
- **Structured storage** in SQLite makes the semantic layer queryable and reusable
- **LLMs assist search** by expanding queries with related keywords based on context

**Core Principle:** kbindex is a data tool, not an AI. It stores and retrieves semantic information; LLMs provide the intelligence for generation, maintenance, and search decisions.

## The Problem & Solution

**The Challenge:**
- Manual keyword maintenance is tedious and error-prone
- Users often don't know the "right" keyword to find what they need
- Traditional full-text search returns too many irrelevant results

**The Solution:**
kbindex creates a collaborative system between humans, LLMs, and structured data:

1. **LLM generates keywords** - Claude Code (or Ollama/Gemini) automatically extracts comprehensive keywords from documents
2. **Structured storage** - SQLite stores documents, keywords, and their semantic relationships
3. **Relationship network** - Similarity mappings (abbreviations, synonyms, related terms) grow over time
4. **AI-powered search** - LLMs help users discover related keywords, filter by context, and make intelligent search decisions

**The Result:**
Over time, the knowledge base becomes smarter:
- More keywords → better coverage
- More similarities → better discovery
- Context awareness → more precise results

The system builds a **domain-specific semantic layer** that LLMs can leverage for intelligent search, eliminating manual maintenance while keeping the indexer itself simple and deterministic.

## Features

- **AI-Powered Keyword Generation** - Automatic keyword extraction using Claude Code CLI (primary), with Ollama/Gemini fallback
- **Strategic LLM Usage** - Claude Code for quality (keyword generation), Ollama for cost efficiency (context matching)
- **Intelligent Sync** - Automated detection of new/modified documents with keyword regeneration
- **Document Indexing** - Store markdown documents with titles, summaries, and keywords
- **Keyword Management** - Normalized keyword storage with optional categorization
- **Semantic Relationships** - Define similarities between keywords with context and scoring
- **AI-Friendly Output** - JSON format with rich metadata for AI decision-making
- **Simple CLI** - Easy-to-use command-line interface
- **Minimal Dependencies** - Core functionality uses stdlib only; optional LLM backends for keyword generation

## Quick Start

**Automated Setup** (recommended):

```bash
# Run setup script to detect and configure AI backends
./scripts/setup.sh

# Or run in interactive mode for guided setup
./scripts/setup.sh --interactive
```

The setup script will:
- Check for required tools (Python, Claude Code, Ollama)
- Verify AI backend availability and configuration
- Initialize database (interactive mode)
- Set up knowledge-base directory (interactive mode)
- Install Python packages (interactive mode)
- Pull Ollama models (interactive mode)
- Provide personalized setup instructions

**Manual Setup**:

Place your markdown documents in `./knowledge-base/` directory (or create a symlink to your existing knowledge base).

```bash
# Initialize database
./kbindex.py init

# Sync all documents from knowledge-base/ directory
./scripts/sync_kb.sh

# Search for documents
./kbindex.py search "reinforcement learning" --format json

# Find similar keywords
./kbindex.py similar "reinforcement learning" --format json

# Get detailed help for any command
./kbindex.py search --help
./kbindex.py similar --help
```

## Installation

### Requirements

- Python 3.8 or higher
- One of the following AI backends (at least one recommended):
  - **Claude Code CLI** (recommended for keyword generation)
  - **Ollama** (recommended for free, local context matching)
  - **Google Gemini API** (cloud alternative, requires API key)

### Quick Setup

```bash
cd kb-indexer
chmod +x scripts/setup.sh

# Run interactive setup (recommended)
./scripts/setup.sh --interactive

# Or run detection only
./scripts/setup.sh
```

The interactive setup will guide you through:
1. Database initialization
2. Knowledge-base directory setup
3. Python package installation (ollama, google-genai)
4. Ollama model download (llama3.2:3b)

### Manual Installation

```bash
# Make scripts executable
chmod +x kbindex.py scripts/*.sh

# Initialize database
./kbindex.py init

# Set up knowledge-base directory
mkdir knowledge-base
# OR link to existing directory
ln -s /path/to/your/knowledge-base knowledge-base

# Install optional Python packages for AI features
pip install ollama google-genai

# Pull Ollama model (if using Ollama)
ollama pull llama3.2:3b

# Set up Gemini API key (if using Gemini)
# Get your API key from: https://aistudio.google.com/apikey
export GEMINI_API_KEY='your-api-key-here'
# To make it permanent, add the export line to ~/.bashrc or ~/.zshrc
```

The CLI provides comprehensive help messages with:
- Detailed descriptions of what each command does
- Examples of valid input values
- Explanation of all options and formats
- Value ranges for numeric parameters

### AI Backend Setup

**Claude Code CLI** (recommended for keyword generation):
- Install from: https://claude.com/claude-code
- No additional configuration needed

**Ollama** (recommended for context matching):
```bash
# Install Ollama from: https://ollama.ai
# Pull the recommended model
ollama pull llama3.2:3b
```

**Google Gemini** (optional cloud alternative):
```bash
# Install Python package
pip install google-genai

# Get API key from: https://aistudio.google.com/apikey
# Set environment variable
export GEMINI_API_KEY='your-api-key-here'

# To make it permanent, add to your shell config:
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or ~/.zshrc
```

## Documentation

- [DESIGN.md](DESIGN.md) - Comprehensive design document
- [schema.sql](schema.sql) - Database schema
- [examples/](examples/) - Example keyword and similarity files

## Usage Examples

### Indexing Documents

```bash
# Sync entire knowledge-base directory (recommended - uses AI to generate keywords)
# Automatically generates/updates .keywords.json files using Claude Code CLI
# Looks for knowledge-base/ in project root, can be a symlink or actual directory
# Can be run from any directory - kbindex.py uses kb_index.db in its own directory
./scripts/sync_kb.sh

# Generate keywords for a single document using AI
./scripts/generate_keywords.py knowledge-base/my-doc.md  # Uses Claude Code by default
./scripts/generate_keywords.py knowledge-base/my-doc.md --backend ollama  # Force Ollama
./scripts/generate_keywords.py knowledge-base/my-doc.md --backend gemini  # Force Gemini

# Add single document (requires .keywords.json file)
./kbindex.py add knowledge-base/my-doc.md

# Update existing document
./kbindex.py update knowledge-base/my-doc.md

# Remove document from index
./kbindex.py remove my-doc.md  # Use filename as stored in database

# List all indexed documents (timestamps shown in local time)
./kbindex.py list-docs

# Show document details (timestamps shown in local time)
./kbindex.py show my-doc.md  # Use filename as stored in database
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
./kbindex.py keywords knowledge-base/my-doc.md

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
      "context": "machine learning and AI research",
      "score": 1.0,
      "directional": false
    }
  ]
}
```

**Note**: The `context` field should describe the domain/background where these keywords are used (e.g., "game AI competitions", "natural language processing"), not explain the relationship (the `type` field already does that).

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

**LLM Backend Strategy:**
The project uses different LLMs strategically to balance quality and cost:
- **Claude Code** (sync_kb.sh): High-quality keyword generation - run once per document, quality matters most
- **Ollama** (similar command default): Cost-free context matching with `llama3.2:3b` - run frequently during searches, efficiency matters most
- **Gemini** (optional): Cloud alternative with `gemini-2.0-flash-exp` requiring `GEMINI_API_KEY` environment variable

All output includes rich context for AI decision-making (similarity types, relevance scores, directional flags, context match scores).

### Suggested AI Agent Instructions

To integrate kb-indexer with your AI agent (like Claude Code), add this section to your project's CLAUDE.md file:

```markdown
## Using kb-indexer for Knowledge Base Search

The `kb-indexer/` tool provides intelligent keyword-based search and discovery for the knowledge base. Use it proactively when the user asks questions that might be answered by existing documentation.

**IMPORTANT:** Always `cd kb-indexer/` before running commands. **Always check `--help`** for detailed usage, especially for option values and guidelines.

**Search Workflow:**
1. **Search** with user's keywords using `search` subcommand
2. **If no results:** Use `similar` subcommand to find indexed keywords (always provide --user-context)
3. **Re-search** with discovered keywords from step 2

**Key Behaviors:**
- Search uses exact keyword matching - does NOT auto-expand to similar keywords
- The `similar` subcommand helps discover what keywords to actually search for

**When kb-indexer search fails but you find document via ls/find/grep:**
1. Check `./kbindex.py relate --help` to see similarity types, score guidelines, and context examples
2. Use the `relate` subcommand with suggested terms from the help message
3. This helps future searches succeed with common search terms
```

This instructs your AI agent to:
- Use kb-indexer proactively for knowledge base queries
- Follow the exact → similar → re-search workflow
- Build the semantic layer by adding relationships when searches fail

## Project Structure

```
kb-indexer/
├── README.md                  # This file
├── DESIGN.md                  # Design documentation
├── IMPLEMENTATION.md          # Implementation details and completed features
├── schema.sql                 # Database schema
├── requirements.txt           # Python dependencies (optional: ollama, google-genai)
├── .gitignore                 # Git ignore (includes knowledge-base/)
├── kbindex.py                 # Main CLI entry point
├── kb_index.db                # SQLite database (created on init)
├── knowledge-base/            # Document directory (symlink or actual dir, git-ignored)
│   ├── doc1.md
│   ├── doc1.keywords.json    # Generated keywords
│   └── ...
├── kb_indexer/
│   ├── __init__.py
│   ├── database.py           # SQLite database operations
│   ├── parser.py             # JSON/markdown parsing
│   ├── search.py             # Query and search operations
│   └── context_matcher.py    # LLM-based context matching (Ollama/Gemini)
├── scripts/
│   ├── sync_kb.sh            # Sync knowledge-base with AI keyword generation
│   └── generate_keywords.py  # AI-powered keyword extraction
├── examples/
│   ├── sample.keywords.json
│   ├── similarities.json
│   ├── ai-llm-vs-reinforcement-learning.keywords.json    # Real test data
│   ├── building-rag-systems-python.keywords.json        # Real test data
│   └── python-pip-to-uv-modern-project-management.keywords.json  # Real test data
└── tests/
    ├── test_database.py       # Database operations
    ├── test_parser.py         # JSON/markdown parsing
    ├── test_search.py         # Search engine
    └── test_real_data.py      # Tests using real keywords.json files
```

**Note**: The `knowledge-base/` directory can be a symlink to your existing knowledge base or an actual directory. It's git-ignored to prevent accidental commits.

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

All 59 tests pass:
- 16 database operation tests
- 12 parser tests (keywords, similarities, markdown)
- 12 search engine tests
- 19 real data tests (validation, database operations, similarity, integrity, search)

Real data tests validate the system using actual keywords.json files from the knowledge base:
- `ai-llm-vs-reinforcement-learning.keywords.json` (37 keywords)
- `building-rag-systems-python.keywords.json` (31 keywords)
- `python-pip-to-uv-modern-project-management.keywords.json` (37 keywords)

### Code Organization

- **database.py** (600+ lines) - Complete SQLite CRUD operations
- **parser.py** (250+ lines) - JSON and markdown parsing with validation
- **search.py** (250+ lines) - Search, filtering, and output formatting
- **kbindex.py** (600+ lines) - CLI with all commands and detailed help messages
- **context_matcher.py** (150+ lines) - LLM-based context matching for similarity filtering

## License

MIT License - see [LICENSE](LICENSE) file for details

## See Also

- [DESIGN.md](DESIGN.md) - Detailed design decisions and architecture
- [knowledge-base/](knowledge-base/) - Document repository
