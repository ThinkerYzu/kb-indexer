#!/bin/bash
# Sync knowledge base documents to kb-indexer database
# This script checks ../knowledge-base/ directory and:
# - Generates/updates .keywords.json files for documents using AI (if document changed)
# - Adds new documents that aren't in the database
# - Updates documents that have been modified (file mtime > db updated_at)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_INDEXER_DIR="$(dirname "$SCRIPT_DIR")"
# Resolve KB_DIR to absolute path
if [ -n "$KB_DIR" ]; then
    KB_DIR="$(cd "$KB_DIR" 2>/dev/null && pwd)" || KB_DIR="${KB_DIR}"
else
    # First try kb-indexer/knowledge-base/, then fall back to ../knowledge-base/
    if [ -d "$KB_INDEXER_DIR/knowledge-base" ]; then
        KB_DIR="$(cd "$KB_INDEXER_DIR/knowledge-base" 2>/dev/null && pwd)" || KB_DIR="$KB_INDEXER_DIR/knowledge-base"
    else
        KB_DIR="$(cd "$(dirname "$KB_INDEXER_DIR")/knowledge-base" 2>/dev/null && pwd)" || KB_DIR="$(dirname "$KB_INDEXER_DIR")/knowledge-base"
    fi
fi
KBINDEX="$KB_INDEXER_DIR/kbindex.py"
KEYGEN="$SCRIPT_DIR/generate_keywords.py"

# Change to kb-indexer directory to ensure database is created/used there
cd "$KB_INDEXER_DIR"

echo "=== Knowledge Base Sync Script ==="
echo "KB Indexer: $KB_INDEXER_DIR"
echo "Knowledge Base: $KB_DIR"
echo "Database: $KB_INDEXER_DIR/kb_index.db"
echo ""

# Check if knowledge-base directory exists
if [ ! -d "$KB_DIR" ]; then
    echo "Error: Knowledge base directory not found: $KB_DIR"
    echo "Set KB_DIR environment variable to specify custom location"
    exit 1
fi

# Check if kbindex.py exists
if [ ! -x "$KBINDEX" ]; then
    echo "Error: kbindex.py not found or not executable: $KBINDEX"
    exit 1
fi

# Check if generate_keywords.py exists
if [ ! -x "$KEYGEN" ]; then
    echo "Error: generate_keywords.py not found or not executable: $KEYGEN"
    exit 1
fi

# Get list of currently indexed documents with their updated_at timestamps (in local time)
echo "Fetching currently indexed documents..."
INDEXED_DATA=$("$KBINDEX" list-docs --format json 2>/dev/null)

# Process all markdown files in knowledge-base
echo "Scanning knowledge-base for markdown files..."
ADDED=0
UPDATED=0
SKIPPED=0
UNCHANGED=0
GENERATED=0

while read -r MD_FILE; do
    # Get just the filename (basename)
    FILENAME=$(basename "$MD_FILE")

    # Get markdown file modification time
    MD_MTIME=$(stat -c %Y "$MD_FILE" 2>/dev/null || stat -f %m "$MD_FILE" 2>/dev/null)

    # Check if document is already indexed and get its updated_at
    # list-docs returns local time, so parse it as local time
    DB_UPDATED=$(echo "$INDEXED_DATA" | python3 -c "
import json, sys, datetime
try:
    data = json.load(sys.stdin)
    for doc in data:
        if doc['filepath'] == '$FILENAME':
            # Parse updated_at timestamp as local time (format: 2025-10-05 16:09:42)
            dt = datetime.datetime.strptime(doc['updated_at'], '%Y-%m-%d %H:%M:%S')
            print(int(dt.timestamp()))
            break
except:
    pass
" 2>/dev/null)

    # Check for corresponding keywords file
    KEYWORDS_FILE="${MD_FILE%.md}.keywords.json"

    # Determine what action to take
    if [ -z "$DB_UPDATED" ]; then
        # Document doesn't exist in database
        # Check if keywords file exists, generate if needed
        if [ ! -f "$KEYWORDS_FILE" ]; then
            echo "🤖 GENERATE: $FILENAME (missing keywords)"
            if ! "$KEYGEN" "$MD_FILE" >/dev/null 2>&1; then
                echo "   ❌ Failed to generate keywords for $FILENAME"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            GENERATED=$((GENERATED + 1))
        fi

        # Add to database
        echo "➕ ADD: $FILENAME"
        "$KBINDEX" add "$MD_FILE" --keywords "$KEYWORDS_FILE"
        ADDED=$((ADDED + 1))

    elif [ "$MD_MTIME" -gt "$DB_UPDATED" ]; then
        # Markdown file is newer than database - need to update

        # Check if keywords file exists and if it needs regeneration
        if [ ! -f "$KEYWORDS_FILE" ]; then
            echo "🤖 GENERATE: $FILENAME (missing keywords)"
            if ! "$KEYGEN" "$MD_FILE" >/dev/null 2>&1; then
                echo "   ❌ Failed to generate keywords for $FILENAME"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            GENERATED=$((GENERATED + 1))
        else
            # Check if markdown is newer than keywords file
            KW_MTIME=$(stat -c %Y "$KEYWORDS_FILE" 2>/dev/null || stat -f %m "$KEYWORDS_FILE" 2>/dev/null)
            if [ "$MD_MTIME" -gt "$KW_MTIME" ]; then
                echo "🤖 REGENERATE: $FILENAME (document modified)"
                if ! "$KEYGEN" "$MD_FILE" >/dev/null 2>&1; then
                    echo "   ❌ Failed to regenerate keywords for $FILENAME"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                GENERATED=$((GENERATED + 1))
            fi
        fi

        # Update database
        echo "🔄 UPDATE: $FILENAME (file modified)"
        "$KBINDEX" update "$MD_FILE" --keywords "$KEYWORDS_FILE"
        UPDATED=$((UPDATED + 1))

    else
        # Markdown file is older than or same as database - skip
        echo "✓ UNCHANGED: $FILENAME"
        UNCHANGED=$((UNCHANGED + 1))
    fi
done < <(find -L "$KB_DIR" -type f -name "*.md")

echo ""
echo "=== Sync Complete ==="
echo "Keywords Generated: $GENERATED"
echo "Documents Added: $ADDED"
echo "Documents Updated: $UPDATED"
echo "Documents Unchanged: $UNCHANGED"
echo "Documents Skipped: $SKIPPED (failed to generate keywords)"
