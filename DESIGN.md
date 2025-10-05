# Knowledge Base Indexer - Design Document

## Overview

**kbindex** is a pure data tool for indexing and searching documents in the `knowledge-base/` directory. It provides structured data about documents, keywords, and their relationships for AI agents to consume and make intelligent search decisions.

**Core Principle**: kbindex stores and retrieves data. AI agents make all decisions about search expansion and relevance.

---

## Architecture

### Components

1. **SQLite Database** - Stores documents, keywords, and relationships
2. **CLI Interface** - Command-line tool for data operations
3. **JSON Data Format** - Human-editable keyword and similarity definitions
4. **Python Library** - Core functionality for database operations

### Data Flow

```
knowledge-base/
├── doc.md
└── doc.keywords.json  ──┐
                         │
                         ├──> kbindex CLI ──> SQLite DB
                         │
similarities.json  ──────┘

AI Agent ──> kbindex query ──> Structured JSON ──> AI decision ──> Final search
```

---

## Database Schema

### Tables

#### 1. `documents`
Stores metadata about indexed documents.

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,           -- Relative path from knowledge-base/
    title TEXT,                              -- Extracted from markdown H1 or provided
    summary TEXT,                            -- Short description of document
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes:**
- `filepath` (unique)

#### 2. `keywords`
Stores normalized keywords.

```sql
CREATE TABLE keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT UNIQUE NOT NULL,            -- Normalized (lowercase, trimmed)
    category TEXT,                           -- Optional: primary, concepts, people, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes:**
- `keyword` (unique)

**Notes:**
- Keywords are normalized to lowercase for consistency
- Category is optional grouping (e.g., "primary", "ai_ml_concepts", "people")

#### 3. `document_keywords`
Many-to-many relationship between documents and keywords.

```sql
CREATE TABLE document_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    keyword_id INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE,
    UNIQUE(document_id, keyword_id)
);
```

**Indexes:**
- `document_id`
- `keyword_id`
- Composite unique constraint on (document_id, keyword_id)

#### 4. `keyword_similarities`
Defines relationships between keywords for AI agents to interpret.

```sql
CREATE TABLE keyword_similarities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_id_1 INTEGER NOT NULL,
    keyword_id_2 INTEGER NOT NULL,
    similarity_type TEXT NOT NULL,           -- Type of relationship
    context TEXT NOT NULL,                   -- Human-readable explanation
    score REAL DEFAULT 0.5,                  -- Relevance score (0-1)
    directional BOOLEAN DEFAULT 0,           -- 0=bidirectional, 1=kw1→kw2 only
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (keyword_id_1) REFERENCES keywords(id) ON DELETE CASCADE,
    FOREIGN KEY (keyword_id_2) REFERENCES keywords(id) ON DELETE CASCADE,
    UNIQUE(keyword_id_1, keyword_id_2),
    CHECK(keyword_id_1 < keyword_id_2)       -- Ensure consistent ordering
);
```

**Indexes:**
- `keyword_id_1`
- `keyword_id_2`
- Composite unique constraint on (keyword_id_1, keyword_id_2)

**Notes:**
- Always store with `keyword_id_1 < keyword_id_2` to avoid duplicates
- `directional` flag indicates if relationship is one-way

---

## Similarity Types

Keyword relationships are categorized by type. AI agents interpret these to make search decisions.

### Similarity Type Reference

| Type | Description | Example | AI Interpretation Hint |
|------|-------------|---------|----------------------|
| `synonym` | Exact same meaning | "LLM" ↔ "large language models" | Usually auto-include |
| `abbreviation` | Short form of term | "RL" ↔ "reinforcement learning" | Usually auto-include |
| `related_concept` | Related ideas in same domain | "RL" ↔ "experience learning" | Consider including |
| `broader` | More general concept | "machine learning" ← "reinforcement learning" | Include for general context |
| `narrower` | More specific concept | "reinforcement learning" → "TD learning" | Include for specific examples |
| `contrast` | Opposing concepts | "supervised learning" ↔ "unsupervised learning" | Usually exclude unless comparing |
| `application` | Practical use case | "reinforcement learning" → "AlphaGo" | Include for examples |
| `prerequisite` | Required understanding | "value function" → "TD learning" | Include for foundational understanding |
| `component` | Part of larger concept | "policy" → "reinforcement learning" | Include for detailed breakdown |

**Note:** These are suggestions for AI agents. The tool only stores the data; AI decides how to use it.

---

## File Formats

### Document Keywords JSON

Each document can have a companion `.keywords.json` file.

**Location:** `knowledge-base/<filename>.keywords.json`

**Format:**
```json
{
  "filepath": "ai-llm-vs-reinforcement-learning.md",
  "title": "AI: LLMs vs Reinforcement Learning",
  "summary": "Richard Sutton argues LLMs are dead end for AGI; true intelligence requires experience learning through direct world interaction.",
  "keywords": [
    "reinforcement learning",
    "RL",
    "LLM",
    "large language models",
    "AGI",
    "world models"
  ],
  "categories": {
    "primary": ["reinforcement learning", "LLM", "AGI"],
    "concepts": ["world models", "experience learning"],
    "people": ["Richard Sutton", "Yann LeCun"]
  }
}
```

**Fields:**
- `filepath` (required): Path relative to knowledge-base/
- `title` (optional): Document title (falls back to first H1 in markdown)
- `summary` (required): Brief description of document
- `keywords` (required): Flat list of all keywords
- `categories` (optional): Grouped keywords by type

### Similarities JSON

Global file defining keyword relationships.

**Location:** `kb-indexer/similarities.json`

**Format:**
```json
{
  "similarities": [
    {
      "keyword1": "RL",
      "keyword2": "reinforcement learning",
      "type": "abbreviation",
      "context": "In machine learning literature, RL is the standard abbreviation for reinforcement learning",
      "score": 1.0,
      "directional": false
    },
    {
      "keyword1": "reinforcement learning",
      "keyword2": "experience learning",
      "type": "related_concept",
      "context": "Reinforcement learning is a type of experience learning where agents learn through perception-action-reward cycles",
      "score": 0.9,
      "directional": false
    },
    {
      "keyword1": "reinforcement learning",
      "keyword2": "AlphaGo",
      "type": "application",
      "context": "AlphaGo is a famous implementation demonstrating RL algorithms applied to the game of Go",
      "score": 0.6,
      "directional": true
    }
  ]
}
```

**Fields:**
- `keyword1`, `keyword2` (required): The two related keywords
- `type` (required): Similarity type (see table above)
- `context` (required): Describes the domain/situation where this relationship applies. Helps AI agents understand WHEN to use this similarity for search expansion.
  - Example: "In AI/ML contexts, LLM is the standard abbreviation used interchangeably with 'large language models'"
  - Example: "In machine learning literature, RL refers to reinforcement learning algorithms based on reward signals"
  - Example: "AlphaGo is a famous application of RL algorithms in the game of Go"
- `score` (optional, default 0.5): Relevance score 0-1
- `directional` (optional, default false): If true, relationship is keyword1→keyword2 only

---

## CLI Interface

### Document Operations

```bash
# Add document to index
kbindex add <filepath> [--keywords <keywords.json>]

# Update existing document
kbindex update <filepath>

# Remove document from index
kbindex remove <filepath>

# List all indexed documents
kbindex list-docs [--format json|table]

# Show document details
kbindex show <filepath> [--format json]
```

### Keyword Operations

```bash
# List keywords for a document
kbindex keywords <filepath> [--format json|table]

# List documents containing keyword
kbindex docs <keyword> [--format json|table]

# Get keyword statistics
kbindex stats <keyword> [--format json]
# Returns: document count, related keywords count
```

### Similarity Operations

```bash
# Get all similar keywords with context
kbindex similar <keyword> [--format json]

# Filter by similarity type
kbindex similar <keyword> --type synonym [--format json]
kbindex similar <keyword> --type related_concept [--format json]

# Context-aware similarity search (uses AI to match contexts)
kbindex similar <keyword> --user-context "your context description" [--format json]
kbindex similar <keyword> --user-context "game AI" --context-threshold 0.8

# Choose LLM backend (default: ollama for local, free usage)
kbindex similar <keyword> --user-context "context" --llm-backend ollama  # Local Ollama
kbindex similar <keyword> --user-context "context" --llm-backend gemini  # Cloud Gemini

# Specify custom model
kbindex similar <keyword> --user-context "context" --llm-backend ollama --llm-model llama3.2:3b

# Add similarity relationship
kbindex relate <keyword1> <keyword2> \
  --type <type> \
  --context "explanation" \
  --score 0.8 \
  [--directional]

# Remove similarity relationship
kbindex unrelate <keyword1> <keyword2>

# Import similarities from JSON
kbindex import-similarities <file.json>
```

### Search Operations

```bash
# Search for documents by keyword
kbindex search <keyword> [--format json|table]

# Multiple keywords (OR)
kbindex search <keyword1> <keyword2> --or [--format json]

# Multiple keywords (AND)
kbindex search <keyword1> <keyword2> --and [--format json]
```

### Database Operations

```bash
# Initialize new database
kbindex init [--db <path>]

# Export all data to JSON
kbindex export [--output <file.json>]

# Import data from JSON
kbindex import <file.json>

# Database statistics
kbindex db-stats
# Returns: document count, keyword count, similarity count
```

---

## Output Format for AI Consumption

All commands support `--format json` for machine-readable output.

### Search Results

```json
{
  "query": {
    "keywords": ["reinforcement learning"],
    "mode": "exact"
  },
  "results": [
    {
      "filepath": "ai-llm-vs-reinforcement-learning.md",
      "title": "AI: LLMs vs Reinforcement Learning",
      "summary": "Richard Sutton argues...",
      "matched_keywords": ["reinforcement learning", "RL"]
    }
  ],
  "count": 1
}
```

### Similar Keywords

```json
{
  "keyword": "reinforcement learning",
  "similar_keywords": [
    {
      "keyword": "RL",
      "similarity_type": "abbreviation",
      "context": "RL is abbreviation for reinforcement learning",
      "score": 1.0,
      "directional": false
    },
    {
      "keyword": "experience learning",
      "similarity_type": "related_concept",
      "context": "RL is type of experience learning through perception-action-reward cycle",
      "score": 0.9,
      "directional": false
    }
  ],
  "count": 2
}
```

### Document Details

```json
{
  "filepath": "ai-llm-vs-reinforcement-learning.md",
  "title": "AI: LLMs vs Reinforcement Learning",
  "summary": "Richard Sutton argues LLMs are dead end...",
  "keywords": [
    {
      "keyword": "reinforcement learning",
      "category": "primary"
    },
    {
      "keyword": "LLM",
      "category": "primary"
    }
  ],
  "created_at": "2025-10-04T22:00:00",
  "updated_at": "2025-10-04T22:00:00"
}
```

---

## AI Agent Usage Pattern

### Core Use Case: Finding Related Documents via Similar Keywords

**Problem:** Documents may be relevant but don't contain the exact search keyword.

**Example:**
- User searches for "reinforcement learning"
- Document A has keyword "reinforcement learning" ✓ (direct match)
- Document B has keywords "RL", "experience learning" but NOT "reinforcement learning"
- Document C has keyword "AlphaGo" (application of RL)
- Without similarity expansion: Only Document A is found
- With similarity expansion: Documents A, B, and C are found

**Solution:** AI agent uses similar keywords to discover related documents.

### Example Workflow: Semantic Search Expansion

**Scenario:** User asks "Find documents about reinforcement learning"

#### Step 1: Direct Search
```bash
kbindex search "reinforcement learning" --format json
```

**Result:**
- Document A: "AI: LLMs vs Reinforcement Learning" (has keyword "reinforcement learning")

#### Step 2: Query Similar Keywords
```bash
kbindex similar "reinforcement learning" --format json
```

**AI receives:**
```json
{
  "keyword": "reinforcement learning",
  "similar_keywords": [
    {
      "keyword": "RL",
      "similarity_type": "abbreviation",
      "context": "RL is abbreviation for reinforcement learning",
      "score": 1.0,
      "directional": false
    },
    {
      "keyword": "experience learning",
      "similarity_type": "related_concept",
      "context": "RL is type of experience learning through perception-action-reward cycle",
      "score": 0.9,
      "directional": false
    },
    {
      "keyword": "AlphaGo",
      "similarity_type": "application",
      "context": "AlphaGo is famous implementation of RL algorithms",
      "score": 0.6,
      "directional": true
    },
    {
      "keyword": "supervised learning",
      "similarity_type": "contrast",
      "context": "Different learning paradigm - supervised uses labeled data, RL uses rewards",
      "score": 0.3,
      "directional": false
    }
  ]
}
```

#### Step 3: AI Decides Which Keywords to Include

**Decision logic (AI's responsibility):**
- `"RL"` - abbreviation, score 1.0 → **Include** (synonym, high confidence)
- `"experience learning"` - related_concept, score 0.9 → **Include** (closely related)
- `"AlphaGo"` - application, score 0.6 → **Maybe include** (if user wants examples)
- `"supervised learning"` - contrast, score 0.3 → **Exclude** (opposite concept)

**AI selects:** `["RL", "experience learning"]`

#### Step 4: Expanded Search
```bash
kbindex search "reinforcement learning" "RL" "experience learning" --or --format json
```

**Result:**
- Document A: "AI: LLMs vs Reinforcement Learning" (has "reinforcement learning")
- Document B: "Experience-Based Learning Systems" (has "experience learning", "RL")
- Document C: "Trial and Error Learning" (has "experience learning")

#### Step 5: AI Presents Results

AI can now present a broader set of relevant documents, including those that:
- Use abbreviations instead of full terms
- Discuss related concepts without using the exact keyword
- Are semantically related through similarity relationships

### Key Insight

**Similar keywords enable semantic document discovery:**

| Without Similarity | With Similarity |
|-------------------|-----------------|
| Exact keyword match only | Semantic expansion |
| Misses synonyms | Finds "RL" documents |
| Misses related concepts | Finds "experience learning" documents |
| Misses applications | Finds "AlphaGo" documents (if desired) |
| Limited recall | Improved recall while maintaining precision |

**The AI agent uses `similarity_type`, `context`, and `score` to:**
1. Discover related documents that don't have the exact keyword
2. Decide which similar keywords expand the search appropriately
3. Avoid irrelevant expansions (e.g., contrasting concepts)
4. Balance precision (exact matches) with recall (related documents)

### Context-Aware Similarity Search

**Problem:** Generic similarity search may return keywords from different contexts.

**Example:**
- Query: "RL" in context of "game AI"
- Generic results include:
  - "reinforcement learning" (general ML term) ✓
  - "real-time systems" (different context) ✗
  - "AlphaGo" (game AI application) ✓

**Solution:** Use AI to filter similarities by context relevance.

#### Workflow with Context Filtering

```bash
kbindex similar "RL" --user-context "game AI and competitions" --format json
```

**Process:**
1. Retrieve all similar keywords (normal similarity search)
2. For each similarity:
   - Extract stored context AND similarity_type
   - Use LLM to evaluate: Does the similarity (considering both context and type) match user's context?
   - LLM receives: similarity_type, stored context, user context
   - Calculate context match score (0.0-1.0)
3. Filter by threshold (default: 0.7)
4. Return only contextually relevant similarities

**Why include similarity_type:**
- `abbreviation` in "academic papers" context → highly relevant
- `abbreviation` in "casual conversation" context → may be less relevant
- `application` type helps LLM understand it's a specific use case
- `contrast` type signals opposing concept (usually exclude)

**Output:**
```json
{
  "keyword": "RL",
  "user_context": "game AI and competitions",
  "similar_keywords": [
    {
      "keyword": "AlphaGo",
      "similarity_type": "application",
      "context": "AlphaGo demonstrates RL in game of Go",
      "score": 0.6,
      "context_match_score": 0.92,
      "directional": true
    }
  ],
  "count": 1
}
```

**Benefits:**
- More precise search expansion
- Reduces false positives from ambiguous terms
- Adapts to user's specific domain/use case
- Leverages stored context metadata effectively

---

## Implementation Status

### Phase 1: Core Database & Basic Operations ✅ COMPLETED
- [x] Database schema design
- [x] Database initialization
- [x] Document CRUD operations
- [x] Keyword operations
- [x] Basic search

### Phase 2: Similarity System ✅ COMPLETED
- [x] Similarity table operations
- [x] Import/export similarities
- [x] Similar keyword queries

### Phase 3: CLI Interface ✅ COMPLETED
- [x] Argument parsing
- [x] JSON/table output formatting
- [x] Error handling

### Phase 4: Data Import ✅ COMPLETED
- [x] Parse keywords.json files
- [x] Parse similarities.json files
- [x] Markdown parsing for title extraction

### Phase 5: Testing & Documentation ✅ COMPLETED
- [x] Unit tests (40 tests, all passing)
- [x] Integration tests (CLI tested with example data)
- [x] User documentation (README.md)
- [x] API documentation (inline docstrings)

### Phase 6: Context-Aware Similarity Search ✅ COMPLETED
- [x] Create `kb_indexer/context_matcher.py` module
- [x] Implement LLM-based context matching with multiple backends
  - LLM receives: keyword, related_keyword, similarity_type, stored context, user context
  - Returns: match decision + confidence score
  - **Backends**: Ollama (local, default) and Gemini (cloud)
- [x] Add `--user-context` flag to `similar` command
- [x] Add `--context-threshold` parameter (default: 0.7)
- [x] Add `--llm-backend` parameter (default: ollama)
- [x] Add `--llm-model` parameter for custom model selection
- [x] Include `context_match_score` in JSON output
- [x] Add dependencies: `ollama`, `google-genai`, `python-dotenv`
- [x] Support for local Ollama models (no API key needed)
- [x] Support for cloud Gemini API (requires `GEMINI_API_KEY`)
- [x] Improved prompt with similarity type priority guidance
- [x] Update CLI help documentation
- [x] Add examples to README.md

**Current Status:** All planned features implemented. Context-aware search working with both local and cloud LLM backends.

### Implementation Notes

**Key Bug Fixes:**
1. **Database.add_keyword()** - Fixed `INSERT OR IGNORE` lastrowid behavior
   - Issue: SQLite returns previous lastrowid when row is ignored
   - Solution: Check `cursor.rowcount > 0` instead of `cursor.lastrowid`
   - Location: kb_indexer/database.py:244

2. **Database.add_similarity()** - Fixed CHECK constraint violation
   - Issue: `INSERT OR REPLACE` didn't work with CHECK constraint
   - Solution: Separate check for existing row, UPDATE vs INSERT
   - Location: kb_indexer/database.py:457-489

**Test Coverage:**
- 40 unit tests covering all modules
- All edge cases tested (duplicates, normalization, cascading deletes)
- Integration testing with example data

**File Statistics:**
- kb_indexer/database.py: 650 lines
- kb_indexer/parser.py: 270 lines
- kb_indexer/search.py: 280 lines
- kbindex.py: 560 lines
- tests/: 600+ lines across 3 test files

---

## Future Enhancements

### Other Potential Features

1. **Full-text search** - Search in document summaries
2. **Keyword co-occurrence** - Suggest similarities based on documents sharing keywords
3. **Embedding-based similarity** - Auto-calculate similarity using embeddings
4. **Version tracking** - Track document changes over time
5. **Multi-language support** - Keywords in multiple languages
6. **Tag system** - Additional metadata beyond keywords
7. **REST API** - HTTP interface for remote access
8. **Context caching** - Cache LLM context comparisons for performance

---

## Design Decisions

### Why SQLite?
- Single file database (portable)
- No server required
- Good performance for read-heavy workloads
- Built-in Python support
- ACID compliance

### Why Normalized Keywords?
- Consistent searching (case-insensitive)
- Reduces duplicates ("RL" vs "RL ")
- Easier similarity matching

### Why Separate keywords.json Files?
- Human-editable alongside documents
- Version controlled with documents
- No database required for editing
- Clear ownership of keywords

### Why Score in Similarities?
- AI agents can threshold by relevance
- Flexibility in expansion strategies
- Can prioritize high-score relationships

### Why Context Field?
- Describes the domain/situation where the keyword relationship applies
- Helps AI agents understand WHEN to use this similarity for search expansion
- Example: "In AI/ML contexts" tells the AI this abbreviation is domain-specific
- Human-readable for manual editing and understanding
- Enables semantic decision-making based on contextual relevance

---

## Security & Data Integrity

### Constraints
- Foreign key constraints ensure referential integrity
- Unique constraints prevent duplicates
- Check constraint ensures consistent similarity ordering

### Input Validation
- File paths validated before database insertion
- Keywords normalized before storage
- Similarity scores clamped to [0, 1]

### No Authentication
- Tool operates on local filesystem
- Assumes trusted environment
- Database file permissions control access

---

## Development Guidelines

### Code Style
- Follow PEP 8
- Type hints for all functions
- Docstrings for public APIs

### Error Handling
- Graceful failures with clear error messages
- Exit codes: 0=success, 1=error, 2=usage error
- Log errors to stderr

### Testing
- Unit tests for all database operations
- Integration tests for CLI commands
- Test with sample knowledge-base files

---

## License & Attribution

**License:** MIT (or as per project license)

**Dependencies:**
- Python 3.8+
- sqlite3 (standard library)
- argparse (standard library)
- json (standard library)

No external dependencies for core functionality.
