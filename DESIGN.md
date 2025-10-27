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
      "matched_keywords": ["reinforcement learning", "rl"],
      "user_keywords": ["reinforcement learning"]
    }
  ],
  "count": 1
}
```

**Fields:**
- `matched_keywords`: Normalized keywords from the document that matched the search
- `user_keywords`: Original keywords provided by the user that led to finding this document

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

### Query Results (with Keyword Expansion)

```json
{
  "query": {
    "question": "How does AlphaGo work?",
    "keywords": ["RL", "AlphaGo"],
    "expanded_keywords": ["rl", "reinforcement learning", "q-learning", "alphago"],
    "expansion_map": {
      "rl": ["reinforcement learning", "q-learning"],
      "alphago": []
    },
    "context": "game AI",
    "threshold": 0.7,
    "expand_depth": 1
  },
  "results": [
    {
      "filepath": "alphago-architecture.md",
      "title": "AlphaGo Architecture",
      "summary": "Deep dive into AlphaGo's architecture...",
      "matched_keywords": ["reinforcement learning", "alphago"],
      "user_keywords": ["rl", "alphago"],
      "keyword_expansions": [
        {
          "original": "rl",
          "expanded": "reinforcement learning"
        }
      ],
      "relevance_score": 0.92,
      "reasoning": "Directly discusses AlphaGo's use of RL techniques",
      "source": "keyword_search"
    }
  ],
  "count": 1,
  "keyword_search_count": 1,
  "grep_search_count": 0
}
```

**Fields:**
- `query.expanded_keywords`: All keywords after expansion (includes original + similar keywords)
- `query.expansion_map`: Maps each original keyword to its expanded keywords
- `results[].user_keywords`: Original keywords from user that led to finding this document
- `results[].keyword_expansions`: List showing which user keywords were expanded (original → expanded)
- `results[].matched_keywords`: Actual document keywords that matched
- `results[].source`: "keyword_search" or "grep_search"

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

## Query Refinement and Feedback Loop

### Overview

Users often don't know the perfect keywords on their first try. The **query refinement loop** allows users to:
1. Try a query with initial keywords
2. Refine the query with better keywords if results are poor
3. Mark which documents were actually helpful
4. Periodically run `learn` command to improve index from all feedback
5. System learns from successful keyword → document mappings across all queries

### User Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. Initial Query                                       │
│  ./kbindex.py query "How does AlphaGo work?"           │
│    --keywords "game AI"                                 │
│    --context "board games"                              │
│                                                          │
│  Results: 2 documents, not very relevant                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  2. Check History                                       │
│  ./kbindex.py history                                   │
│                                                          │
│  Output:                                                 │
│  [1] "How does AlphaGo work?" (2 attempts, 1 hour ago) │
│  [2] "What is Q-learning?" (1 attempt, 2 days ago)     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  3. Refine Query with Better Keywords                  │
│  ./kbindex.py refine 1 \                                │
│    --keywords "AlphaGo" "Monte Carlo" "neural networks" │
│                                                          │
│  Results: 5 documents, much better!                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  4. Mark Helpful Documents                              │
│  ./kbindex.py feedback 1 \                              │
│    --helpful alphago-architecture.md \                  │
│    --helpful monte-carlo-tree-search.md                 │
│                                                          │
│  Feedback recorded to database (no learning yet)        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  5. Learn from All Feedback (Run Periodically)          │
│  ./kbindex.py learn                                     │
│                                                          │
│  Analyzing 5 unprocessed feedback entries...            │
│  ✓ Add similarity: "game AI" ↔ "AlphaGo"               │
│  ✓ Add keyword "Monte Carlo" to helpful docs            │
│  Total: 6 suggestions                                   │
│                                                          │
│  Apply? [y/n] y                                         │
│                                                          │
│  Applied 6 suggestions.                                 │
│  Marked 5 feedback entries as processed.                │
│  Future queries benefit from learned relationships!     │
└─────────────────────────────────────────────────────────┘
```

### Database Schema

#### `queries` Table
Stores unique question/context combinations.

```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    context TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose:** Track user queries over time. Multiple attempts at answering the same question are linked to the same query record.

#### `query_attempts` Table
Tracks each attempt at answering a query with different keywords.

```sql
CREATE TABLE query_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    keywords TEXT NOT NULL,                    -- JSON array: ["kw1", "kw2"]
    expand_depth INTEGER DEFAULT 1,
    threshold REAL DEFAULT 0.7,
    result_count INTEGER DEFAULT 0,
    top_results TEXT,                          -- JSON array of top 10 results
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
);
```

**Purpose:** Record each keyword combination tried for a query. This allows:
- Users to see which keyword sets they've already tried
- System to analyze which keywords led to better results
- Learning algorithm to find successful keyword → document patterns

**Fields:**
- `keywords`: JSON array of user-provided keywords (e.g., `["AlphaGo", "Monte Carlo"]`)
- `top_results`: JSON array of result objects with filepath, score, matched_keywords
- `result_count`: Total number of results found

#### `query_feedback` Table
Stores user feedback on document helpfulness.

```sql
CREATE TABLE query_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    attempt_id INTEGER,                        -- Which attempt found this doc
    filepath TEXT NOT NULL,
    helpful BOOLEAN NOT NULL,                  -- 1=helpful, 0=not helpful
    notes TEXT,                                -- Optional user notes
    processed BOOLEAN DEFAULT 0,               -- 0=unprocessed, 1=processed by learn command
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES query_attempts(id) ON DELETE SET NULL,
    UNIQUE(query_id, filepath)
);
```

**Purpose:** Capture which documents actually helped answer the user's question. This is the **ground truth** for learning.

**Fields:**
- `processed`: Tracks whether this feedback has been analyzed by the `learn` command
  - `0` = unprocessed (new feedback)
  - `1` = processed (learned from this feedback)
- Allows `learn` command to efficiently query only new feedback
- Can be reset with `--reprocess` flag to re-analyze

**Key Insight:** When a document is marked helpful for a query, we know:
- The question/context
- The keywords that found it (from attempt)
- The document's existing keywords (from database)
- Gap: missing keywords or similarities

**Indexes:**
```sql
CREATE INDEX idx_query_feedback_processed ON query_feedback(processed);
```

### CLI Commands

#### `query` Command (Enhanced)
Automatically saves query history.

```bash
./kbindex.py query "How does AlphaGo work?" \
  --keywords "game AI" \
  --context "board games" \
  --format json

# Output includes query_id for later refinement
{
  "query_id": 1,
  "query": {...},
  "results": [...],
  ...
}
```

#### `history` Command (New)
Show recent queries and their attempts.

```bash
# List recent queries
./kbindex.py history [--limit 10]

# Show details for specific query
./kbindex.py history 1 --format json

# Output:
{
  "query_id": 1,
  "question": "How does AlphaGo work?",
  "context": "board games",
  "created_at": "2025-10-26 10:00:00",
  "attempts": [
    {
      "attempt_id": 1,
      "keywords": ["game AI"],
      "result_count": 2,
      "created_at": "2025-10-26 10:00:00"
    },
    {
      "attempt_id": 2,
      "keywords": ["AlphaGo", "Monte Carlo", "neural networks"],
      "result_count": 5,
      "created_at": "2025-10-26 11:00:00"
    }
  ],
  "feedback": [
    {
      "filepath": "alphago-architecture.md",
      "helpful": true,
      "attempt_id": 2
    }
  ]
}
```

#### `refine` Command (New)
Retry a previous query with new keywords.

```bash
# Refine by query ID
./kbindex.py refine 1 \
  --keywords "AlphaGo" "Monte Carlo" "neural networks" \
  [--threshold 0.7] \
  [--expand-depth 1]

# Refine by automatically using last query
./kbindex.py refine --last \
  --keywords "AlphaGo" "Monte Carlo"

# Output: Same as query command, but links to existing query record
```

**Behavior:**
- Reuses question and context from original query
- Creates new attempt record
- Allows users to iterate on keyword selection
- System tracks which keyword combinations work best

#### `feedback` Command (New)
Record which documents were helpful or not helpful.

```bash
# Mark documents as helpful
./kbindex.py feedback 1 \
  --helpful alphago-architecture.md \
  --helpful monte-carlo-tree-search.md

# Mark document as not helpful
./kbindex.py feedback 1 \
  --not-helpful some-irrelevant-doc.md

# Add notes
./kbindex.py feedback 1 \
  --helpful alphago-architecture.md \
  --notes "Great explanation of neural network architecture"

# Interactive mode (prompts user to review results)
./kbindex.py feedback 1 --interactive

# View all feedback for a query
./kbindex.py feedback 1 --show
```

**Behavior:**
- Records which documents helped answer the query (writes to database)
- Optionally links to the attempt that found the document
- **Does NOT trigger learning** - feedback is stored for later batch analysis

#### `learn` Command (New)
Analyze all unprocessed feedback and improve the index in batch.

```bash
# Analyze all feedback and automatically apply improvements
./kbindex.py learn

# Show what would be learned without applying (dry-run)
./kbindex.py learn --dry-run
```

**Output:**
```
Analyzing 5 unprocessed feedback entries from 3 queries...

Keyword Gap Analysis:
✓ Add similarity: "Monte Carlo" ↔ "MCTS" (abbreviation, score 0.9)
  Reason: 3 queries searched "Monte Carlo", helpful docs have "MCTS"

✓ Add similarity: "game AI" ↔ "AlphaGo" (related, score 0.7)
  Reason: 2 queries searched "game AI", found helpful AlphaGo docs

Keyword Augmentation:
✓ Add keyword "neural networks" to alphago-architecture.md
  Reason: 2 queries used this keyword to find this helpful doc

✓ Add keyword "Monte Carlo" to monte-carlo-tree-search.md
  Reason: 3 queries used this keyword to find this helpful doc

Pattern Recognition:
✓ Keywords ["AlphaGo", "Monte Carlo"] appear together in 2 successful queries
  Suggest adding similarity between these terms

Summary:
- 3 similarity suggestions
- 2 keyword augmentation suggestions
- 1 pattern-based suggestion
- Total: 6 suggestions

Applied 6 suggestions successfully.
Marked 5 feedback entries as processed.
```

**Behavior:**
- Always analyzes ALL unprocessed feedback (`processed = 0`)
- Finds patterns across multiple queries
- Shows all suggestions
- Automatically applies suggestions (no confirmation needed)
- Marks all processed feedback as `processed = 1`
- `--dry-run`: Shows suggestions without applying or marking as processed

### Learning Algorithm

The `learn` command analyzes all feedback in batch and generates improvement suggestions:

#### 1. Keyword Gap Analysis
Finds common gaps between user search terms and document keywords.

```python
# Analyze across ALL helpful feedback entries:
feedback_entries = [
    (query_keywords=["Monte Carlo"], doc="mcts.md", doc_keywords=["mcts", "tree search"]),
    (query_keywords=["Monte Carlo"], doc="alphago.md", doc_keywords=["mcts", "alphago"]),
    (query_keywords=["Monte Carlo"], doc="ucb.md", doc_keywords=["mcts", "ucb1"]),
]

# Pattern: 3 queries searched "Monte Carlo" but all docs have "mcts"
# Suggestion: Add similarity "Monte Carlo" ↔ "MCTS" (abbreviation, score 0.9)
# Confidence: HIGH (3 occurrences)
```

**Scoring:** More occurrences = higher confidence score

#### 2. Keyword Augmentation
Adds user keywords to documents they successfully found.

```python
# Example from aggregate analysis:
helpful_mappings = [
    (query_keywords=["neural networks"], doc="alphago.md"),
    (query_keywords=["neural networks", "deep learning"], doc="alphago.md"),
]

# Pattern: "neural networks" keyword led to alphago.md 2 times
# Current doc keywords: ["alphago", "mcts", "deep learning"]
# Suggestion: Add "neural networks" to alphago.md
# Confidence: MEDIUM (2 occurrences)
```

**Deduplication:** Only adds if keyword not already present

#### 3. Pattern Recognition
Identifies keyword co-occurrence in successful queries.

```python
# Analyze keyword combinations across queries:
successful_combinations = [
    (query_keywords=["AlphaGo", "Monte Carlo"], result_count=5, helpful_count=3),
    (query_keywords=["AlphaGo", "MCTS"], result_count=4, helpful_count=2),
    (query_keywords=["game AI", "AlphaGo"], result_count=3, helpful_count=2),
]

# Pattern: Keywords often used together
# Suggestion: Add similarity "AlphaGo" ↔ "Monte Carlo" (related, score 0.6)
# Reasoning: Co-occur in 2 successful queries
```

**Threshold:** Only suggest if co-occurrence >= 2 queries

#### 4. Similarity Scoring (Future Enhancement)
Track expansion success rate to adjust similarity scores.

```python
# Track which expansions led to helpful results:
expansion_stats = {
    ("RL", "reinforcement learning"): {"helpful": 8, "total": 10},  # 80% success
    ("RL", "real life"): {"helpful": 1, "total": 10},  # 10% success
}

# Suggestion: Increase score for "RL" ↔ "reinforcement learning"
# Suggestion: Decrease score for "RL" ↔ "real life"
```

**Note:** Requires tracking expansion chains in query attempts (future work)

### Integration with Existing Learning

The feedback loop **complements** existing auto-learning:

| Learning Type | Trigger | What It Learns |
|--------------|---------|----------------|
| **Auto-indexing** | Grep finds unindexed doc | Add doc to index with LLM keywords |
| **Auto-reindexing** | File modified | Update doc keywords intelligently |
| **Smart learning** | Grep succeeds, keyword fails | Add query keywords as similarities |
| **Feedback learning** (NEW) | User marks doc helpful | Add missing keywords/similarities based on what worked |

**Key Difference:** Feedback learning uses **human-validated ground truth** (user confirmed this doc is helpful), making it more accurate than automated heuristics.

### Privacy & Data Retention

- Query history stored locally in SQLite
- No external data transmission
- Users can clear history: `./kbindex.py clear-history [--before date]`
- Feedback is tied to queries but can be anonymized

### Performance Considerations

- History queries indexed by `created_at DESC` for fast recent query retrieval
- Limit top_results to 10 documents per attempt (reduce storage)
- Feedback unique constraint prevents duplicate feedback per query/document
- Learning applied incrementally (not batch reprocessing)

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

## Implementation Status

✅ **Completed (as of commit b21378c)**

### Database Schema
- ✅ `queries` table - Stores unique question/context combinations
- ✅ `query_attempts` table - Records each keyword combination tried
- ✅ `query_feedback` table - Stores user feedback with `processed` flag
- ✅ All indexes and triggers implemented

### Database Methods (`kb_indexer/database.py`)
- ✅ `get_or_create_query()` - Get/create query record
- ✅ `add_query_attempt()` - Record attempt with keywords and results
- ✅ `get_query()` - Retrieve query details
- ✅ `get_recent_queries()` - List recent queries with counts
- ✅ `get_query_attempts()` - Get all attempts for a query
- ✅ `add_feedback()` - Record helpful/not helpful documents
- ✅ `get_feedback()` - Get feedback for a query
- ✅ `get_unprocessed_feedback()` - Get feedback ready for learning
- ✅ `mark_feedback_processed()` - Mark feedback as processed

### Feedback Learning (`kb_indexer/feedback.py`)
- ✅ `FeedbackLearner` class
- ✅ `analyze_feedback()` - Analyzes all unprocessed feedback
- ✅ Keyword gap analysis - Finds common keyword-document mismatches
- ✅ Keyword augmentation - Adds proven keywords to documents
- ✅ Pattern recognition - Identifies keyword co-occurrence
- ✅ `apply_suggestions()` - Applies improvements to database

### Query History Tracking (`kb_indexer/query.py`)
- ✅ Query method saves attempt with `query_id` and `attempt_id`
- ✅ Stores top 10 results for reference
- ✅ Returns `query_id` and `attempt_id` in output

### CLI Commands (`kbindex.py`)
- ✅ `history` - View recent queries and attempts
- ✅ `refine` - Retry query with new keywords
- ✅ `feedback` - Record helpful/not helpful documents
- ✅ `learn` - Batch learning from all unprocessed feedback

### Documentation
- ✅ DESIGN.md - Complete technical design
- ✅ README.md - User workflows and examples
- ✅ CLAUDE.md - Developer instructions

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
