#!/bin/bash
# Sync knowledge base documents to kb-indexer database
# This script checks ../knowledge-base/ directory and:
# - Adds new documents that aren't in the database
# - Updates documents that have been modified (file mtime > db updated_at)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_INDEXER_DIR="$(dirname "$SCRIPT_DIR")"
KB_DIR="${KB_DIR:-$(dirname "$KB_INDEXER_DIR")/knowledge-base}"
KBINDEX="$KB_INDEXER_DIR/kbindex.py"

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

# Get list of currently indexed documents with their updated_at timestamps
echo "Fetching currently indexed documents..."
INDEXED_DATA=$("$KBINDEX" list-docs --format json 2>/dev/null)

# Process all markdown files in knowledge-base
echo "Scanning knowledge-base for markdown files..."
ADDED=0
UPDATED=0
SKIPPED=0
UNCHANGED=0

find "$KB_DIR" -type f -name "*.md" | while read -r MD_FILE; do
    # Get just the filename (basename)
    FILENAME=$(basename "$MD_FILE")
    
    # Check for corresponding keywords file
    KEYWORDS_FILE="${MD_FILE%.md}.keywords.json"
    
    if [ ! -f "$KEYWORDS_FILE" ]; then
        echo "⚠️  SKIP: $FILENAME (no keywords file)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Get file modification time
    FILE_MTIME=$(stat -c %Y "$MD_FILE" 2>/dev/null || stat -f %m "$MD_FILE" 2>/dev/null)
    
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
echo "Added: $ADDED"
echo "Updated: $UPDATED"
echo "Unchanged: $UNCHANGED"
echo "Skipped: $SKIPPED (missing keywords)"
