#!/bin/bash
# Sync knowledge base documents to kb-indexer database
# This script checks ../knowledge-base/ directory and:
# - Generates/updates .keywords.json files for documents using AI (if document changed)
# - Adds new documents that aren't in the database
# - Updates documents that have been modified (file mtime > db updated_at)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_INDEXER_DIR="$(dirname "$SCRIPT_DIR")"
KB_DIR="${KB_DIR:-$(dirname "$KB_INDEXER_DIR")/knowledge-base}"
KBINDEX="$KB_INDEXER_DIR/kbindex.py"
KEYGEN="$SCRIPT_DIR/generate_keywords.py"

echo "=== Knowledge Base Sync Script ==="
echo "KB Indexer: $KB_INDEXER_DIR"
echo "Knowledge Base: $KB_DIR"
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

# Get list of currently indexed documents with their updated_at timestamps
echo "Fetching currently indexed documents..."
INDEXED_DATA=$("$KBINDEX" list-docs --format json 2>/dev/null)

# Process all markdown files in knowledge-base
echo "Scanning knowledge-base for markdown files..."
ADDED=0
UPDATED=0
SKIPPED=0
UNCHANGED=0
GENERATED=0

find "$KB_DIR" -type f -name "*.md" | while read -r MD_FILE; do
    # Get just the filename (basename)
    FILENAME=$(basename "$MD_FILE")

    # Check for corresponding keywords file
    KEYWORDS_FILE="${MD_FILE%.md}.keywords.json"

    # Get markdown file modification time
    MD_MTIME=$(stat -c %Y "$MD_FILE" 2>/dev/null || stat -f %m "$MD_FILE" 2>/dev/null)

    # Check if keywords file exists
    if [ ! -f "$KEYWORDS_FILE" ]; then
        # Generate new keywords file
        echo "🤖 GENERATE: $FILENAME (missing keywords)"
        if "$KEYGEN" "$MD_FILE" >/dev/null 2>&1; then
            GENERATED=$((GENERATED + 1))
        else
            echo "   ❌ Failed to generate keywords for $FILENAME"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    else
        # Check if markdown is newer than keywords file
        KW_MTIME=$(stat -c %Y "$KEYWORDS_FILE" 2>/dev/null || stat -f %m "$KEYWORDS_FILE" 2>/dev/null)

        if [ "$MD_MTIME" -gt "$KW_MTIME" ]; then
            # Markdown changed, regenerate keywords
            echo "🤖 REGENERATE: $FILENAME (document modified)"
            if "$KEYGEN" "$MD_FILE" >/dev/null 2>&1; then
                GENERATED=$((GENERATED + 1))
            else
                echo "   ❌ Failed to regenerate keywords for $FILENAME"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    # Now process the document for database sync
    # Use the markdown file mtime for comparison (not keywords file)
    FILE_MTIME="$MD_MTIME"

    # Check if document is already indexed and get its updated_at
    # The database stores UTC time, so we need to parse it as UTC
    DB_UPDATED=$(echo "$INDEXED_DATA" | python3 -c "
import json, sys, datetime
try:
    data = json.load(sys.stdin)
    for doc in data:
        if doc['filepath'] == '$FILENAME':
            # Parse updated_at timestamp as UTC (format: 2025-10-05 16:09:42)
            dt = datetime.datetime.strptime(doc['updated_at'], '%Y-%m-%d %H:%M:%S')
            # Treat it as UTC and convert to local timestamp
            dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
            print(int(dt_utc.timestamp()))
            break
except:
    pass
" 2>/dev/null)
    
    if [ -z "$DB_UPDATED" ]; then
        # Document doesn't exist - add it
        echo "➕ ADD: $FILENAME"
        "$KBINDEX" add "$MD_FILE" --keywords "$KEYWORDS_FILE"
        ADDED=$((ADDED + 1))
    elif [ "$FILE_MTIME" -gt "$DB_UPDATED" ]; then
        # File is newer than database - update it
        echo "🔄 UPDATE: $FILENAME (file modified)"
        "$KBINDEX" update "$MD_FILE" --keywords "$KEYWORDS_FILE"
        UPDATED=$((UPDATED + 1))
    else
        # File hasn't changed
        echo "✓ UNCHANGED: $FILENAME"
        UNCHANGED=$((UNCHANGED + 1))
    fi
done

echo ""
echo "=== Sync Complete ==="
echo "Keywords Generated: $GENERATED"
echo "Documents Added: $ADDED"
echo "Documents Updated: $UPDATED"
echo "Documents Unchanged: $UNCHANGED"
echo "Documents Skipped: $SKIPPED (failed to generate keywords)"
