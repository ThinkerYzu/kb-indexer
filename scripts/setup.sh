#!/bin/bash
# Setup script for kb-indexer - detects and configures AI backends

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse command-line arguments
INTERACTIVE=0
if [ "$1" = "--interactive" ] || [ "$1" = "-i" ]; then
    INTERACTIVE=1
fi

echo "=== kb-indexer Setup ==="
echo

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    echo "  Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo

# Check Claude Code CLI
echo "Checking Claude Code CLI..."
if command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Claude Code CLI found ($CLAUDE_VERSION)${NC}"
    HAS_CLAUDE=1
else
    echo -e "${YELLOW}✗ Claude Code CLI not found${NC}"
    echo "  Claude Code is recommended for high-quality keyword generation"
    echo "  Install from: https://claude.com/claude-code"
    HAS_CLAUDE=0
fi
echo

# Check Ollama
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
    echo -e "${GREEN}✓ Ollama found ($OLLAMA_VERSION)${NC}"
    HAS_OLLAMA=1

    # Check if llama3.2:3b model is available
    echo "  Checking for llama3.2:3b model..."
    if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
        echo -e "  ${GREEN}✓ llama3.2:3b model found${NC}"
        HAS_LLAMA_MODEL=1
    else
        echo -e "  ${YELLOW}✗ llama3.2:3b model not found${NC}"
        echo "  This model is used for cost-free context matching"
        HAS_LLAMA_MODEL=0
    fi
else
    echo -e "${YELLOW}✗ Ollama not found${NC}"
    echo "  Ollama provides free, local LLM for context matching"
    echo "  Install from: https://ollama.ai"
    HAS_OLLAMA=0
    HAS_LLAMA_MODEL=0
fi
echo

# Check Python dependencies
echo "Checking Python dependencies..."
MISSING_DEPS=()

# Check for pytest (optional, for development)
if python3 -c "import pytest" 2>/dev/null; then
    echo -e "${GREEN}✓ pytest found (development)${NC}"
else
    echo -e "${YELLOW}✗ pytest not found (optional, for testing)${NC}"
fi

# Check for ollama Python package (optional)
if python3 -c "import ollama" 2>/dev/null; then
    echo -e "${GREEN}✓ ollama Python package found${NC}"
    HAS_OLLAMA_PY=1
else
    echo -e "${YELLOW}✗ ollama Python package not found${NC}"
    MISSING_DEPS+=("ollama")
    HAS_OLLAMA_PY=0
fi

# Check for google-genai (optional)
if python3 -c "import google.genai" 2>/dev/null; then
    echo -e "${GREEN}✓ google-genai package found${NC}"
    HAS_GENAI=1
else
    echo -e "${YELLOW}✗ google-genai package not found${NC}"
    MISSING_DEPS+=("google-genai")
    HAS_GENAI=0
fi
echo

# Check Gemini API key
echo "Checking Gemini API key..."
if [ -n "$GEMINI_API_KEY" ]; then
    echo -e "${GREEN}✓ GEMINI_API_KEY environment variable is set${NC}"
    HAS_GEMINI_KEY=1
else
    echo -e "${YELLOW}✗ GEMINI_API_KEY not set${NC}"
    echo "  Gemini is optional - only needed if you want to use Google's cloud LLM"
    HAS_GEMINI_KEY=0
fi
echo

# Check database
echo "Checking database..."
DB_FILE="$PROJECT_DIR/kb_index.db"
if [ -f "$DB_FILE" ]; then
    echo -e "${GREEN}✓ Database exists: kb_index.db${NC}"
    HAS_DB=1
else
    echo -e "${YELLOW}✗ Database not initialized${NC}"
    HAS_DB=0

    # Interactive mode - ask user
    if [ $INTERACTIVE -eq 1 ]; then
        echo
        read -p "Would you like to initialize the database now? (y/n): " -r INIT_DB
        if [[ $INIT_DB =~ ^[Yy]$ ]]; then
            SHOULD_INIT=1
        else
            SHOULD_INIT=0
        fi
    else
        # Non-interactive mode - auto-initialize
        SHOULD_INIT=1
    fi

    if [ $SHOULD_INIT -eq 1 ]; then
        if [ -f "$PROJECT_DIR/kbindex.py" ]; then
            echo "Initializing database..."
            cd "$PROJECT_DIR"
            if ./kbindex.py init; then
                echo -e "${GREEN}✓ Database initialized successfully${NC}"
                HAS_DB=1
            else
                echo -e "${RED}✗ Failed to initialize database${NC}"
            fi
        else
            echo -e "${RED}✗ kbindex.py not found${NC}"
            echo "  Make sure you're running this script from the kb-indexer directory"
        fi
    fi
fi
echo

# Check knowledge-base directory
echo "Checking knowledge-base directory..."
KB_DIR="$PROJECT_DIR/knowledge-base"
if [ -e "$KB_DIR" ]; then
    if [ -L "$KB_DIR" ]; then
        TARGET=$(readlink -f "$KB_DIR" 2>/dev/null || readlink "$KB_DIR")
        echo -e "${GREEN}✓ knowledge-base/ is a symlink to: $TARGET${NC}"
        HAS_KB_DIR=1
    elif [ -d "$KB_DIR" ]; then
        echo -e "${GREEN}✓ knowledge-base/ directory exists${NC}"
        HAS_KB_DIR=1
    else
        echo -e "${RED}✗ knowledge-base/ exists but is not a directory or symlink${NC}"
        HAS_KB_DIR=0
    fi
else
    echo -e "${YELLOW}✗ knowledge-base/ directory not found${NC}"
    HAS_KB_DIR=0

    # Interactive setup for knowledge-base
    if [ $INTERACTIVE -eq 1 ]; then
        echo
        read -p "Would you like to set up the knowledge-base directory now? (y/n): " -r SETUP_KB
        if [[ $SETUP_KB =~ ^[Yy]$ ]]; then
            echo
            echo "Choose an option:"
            echo "  1) Create a new empty directory"
            echo "  2) Create a symlink to an existing knowledge base"
            echo
            read -p "Enter your choice (1 or 2): " -r KB_CHOICE
            echo

            if [ "$KB_CHOICE" = "1" ]; then
                mkdir -p "$KB_DIR"
                echo -e "${GREEN}✓ Created knowledge-base/ directory${NC}"
                HAS_KB_DIR=1
            elif [ "$KB_CHOICE" = "2" ]; then
                read -p "Enter the path to your existing knowledge base: " KB_PATH
                if [ -d "$KB_PATH" ]; then
                    ln -s "$KB_PATH" "$KB_DIR"
                    echo -e "${GREEN}✓ Created symlink to $KB_PATH${NC}"
                    HAS_KB_DIR=1
                else
                    echo -e "${RED}✗ Directory not found: $KB_PATH${NC}"
                    echo -e "${YELLOW}  You can create the symlink manually later with:${NC}"
                    echo -e "${YELLOW}  ln -s /path/to/your/knowledge-base $KB_DIR${NC}"
                fi
            else
                echo -e "${YELLOW}Invalid choice. Skipping knowledge-base setup.${NC}"
            fi
        fi
    fi
fi
echo

# Summary and recommendations
echo "=== Setup Summary ==="
echo

READY=1

# Keyword generation backend
echo "Keyword Generation (quality-focused, run once per document):"
if [ $HAS_CLAUDE -eq 1 ]; then
    echo -e "  ${GREEN}✓ Claude Code CLI available (recommended)${NC}"
elif [ $HAS_OLLAMA -eq 1 ] && [ $HAS_OLLAMA_PY -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Ollama available (fallback)${NC}"
    echo -e "    ${YELLOW}Claude Code CLI recommended for better quality${NC}"
elif [ $HAS_GENAI -eq 1 ] && [ $HAS_GEMINI_KEY -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Gemini available (fallback)${NC}"
    echo -e "    ${YELLOW}Claude Code CLI recommended for better quality${NC}"
else
    echo -e "  ${RED}✗ No keyword generation backend available${NC}"
    READY=0
fi
echo

# Context matching backend
echo "Context Matching (efficiency-focused, run frequently during searches):"
if [ $HAS_OLLAMA -eq 1 ] && [ $HAS_OLLAMA_PY -eq 1 ]; then
    if [ $HAS_LLAMA_MODEL -eq 1 ]; then
        echo -e "  ${GREEN}✓ Ollama with llama3.2:3b available (recommended)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Ollama available but llama3.2:3b model missing${NC}"
    fi
elif [ $HAS_GENAI -eq 1 ] && [ $HAS_GEMINI_KEY -eq 1 ]; then
    echo -e "  ${YELLOW}⚠ Gemini available (cloud alternative)${NC}"
else
    echo -e "  ${YELLOW}⚠ No context matching backend available${NC}"
    echo -e "    ${YELLOW}Context-aware search (--user-context) will not work${NC}"
fi
echo

# Installation recommendations
if [ $READY -eq 0 ] || [ ${#MISSING_DEPS[@]} -gt 0 ] || [ $HAS_LLAMA_MODEL -eq 0 ] || [ $HAS_KB_DIR -eq 0 ]; then
    echo "=== Recommended Actions ==="
    echo

    ACTION_NUM=1

    if [ $HAS_KB_DIR -eq 0 ]; then
        echo "$ACTION_NUM. Set up knowledge-base directory:"
        echo "   Option A - Create new directory:"
        echo "     mkdir $KB_DIR"
        echo
        echo "   Option B - Link to existing knowledge base:"
        echo "     ln -s /path/to/your/existing/knowledge-base $KB_DIR"
        echo
        ACTION_NUM=$((ACTION_NUM + 1))
    fi

    if [ $HAS_CLAUDE -eq 0 ]; then
        echo "$ACTION_NUM. Install Claude Code CLI (recommended):"
        echo "   Visit: https://claude.com/claude-code"
        echo
        ACTION_NUM=$((ACTION_NUM + 1))
    fi

    if [ $HAS_OLLAMA -eq 0 ]; then
        echo "$ACTION_NUM. Install Ollama (recommended for free local LLM):"
        echo "   Visit: https://ollama.ai"
        echo
        ACTION_NUM=$((ACTION_NUM + 1))
    fi

    if [ $HAS_OLLAMA -eq 1 ] && [ $HAS_LLAMA_MODEL -eq 0 ]; then
        echo "$ACTION_NUM. Pull llama3.2:3b model:"
        echo "   ollama pull llama3.2:3b"
        echo

        # Interactive mode - offer to pull model
        if [ $INTERACTIVE -eq 1 ]; then
            read -p "   Would you like to pull this model now? (y/n): " -r PULL_MODEL
            if [[ $PULL_MODEL =~ ^[Yy]$ ]]; then
                echo "   Pulling llama3.2:3b model (this may take a few minutes)..."
                if ollama pull llama3.2:3b; then
                    echo -e "   ${GREEN}✓ Model pulled successfully${NC}"
                    HAS_LLAMA_MODEL=1
                else
                    echo -e "   ${RED}✗ Failed to pull model${NC}"
                fi
                echo
            fi
        fi

        ACTION_NUM=$((ACTION_NUM + 1))
    fi

    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "$ACTION_NUM. Install optional Python packages:"
        echo "   pip install ${MISSING_DEPS[*]}"
        echo

        # Interactive mode - offer to install packages
        if [ $INTERACTIVE -eq 1 ]; then
            read -p "   Would you like to install these packages now? (y/n): " -r INSTALL_DEPS
            if [[ $INSTALL_DEPS =~ ^[Yy]$ ]]; then
                echo "   Installing packages..."
                if pip install ${MISSING_DEPS[*]}; then
                    echo -e "   ${GREEN}✓ Packages installed successfully${NC}"
                    # Update flags
                    for dep in "${MISSING_DEPS[@]}"; do
                        if [ "$dep" = "ollama" ]; then
                            HAS_OLLAMA_PY=1
                        elif [ "$dep" = "google-genai" ]; then
                            HAS_GENAI=1
                        fi
                    done
                else
                    echo -e "   ${RED}✗ Failed to install packages${NC}"
                fi
                echo
            fi
        fi

        ACTION_NUM=$((ACTION_NUM + 1))
    fi

    if [ $HAS_GENAI -eq 0 ] && [ $HAS_OLLAMA -eq 0 ]; then
        echo "$ACTION_NUM. (Optional) Set up Gemini API:"
        echo "   - Get API key from: https://aistudio.google.com/apikey"
        echo "   - Install: pip install google-genai"
        echo "   - Set: export GEMINI_API_KEY='your-key-here'"
        echo
    fi
fi

# Quick start guide
echo "=== Quick Start ==="
echo

STEP=1
if [ $HAS_DB -eq 0 ]; then
    echo "$STEP. Initialize database:"
    echo "   cd $PROJECT_DIR"
    echo "   ./kbindex.py init"
    echo
    STEP=$((STEP + 1))
fi

if [ $HAS_KB_DIR -eq 1 ]; then
    echo "$STEP. Add your markdown documents to knowledge-base/ directory"
else
    echo "$STEP. Set up knowledge-base/ directory (see recommended actions above)"
fi
echo
STEP=$((STEP + 1))

if [ $HAS_CLAUDE -eq 1 ] || [ $HAS_OLLAMA -eq 1 ] || [ $HAS_GENAI -eq 1 ]; then
    echo "$STEP. Sync documents (generate keywords with AI):"
    echo "   ./scripts/sync_kb.sh"
else
    echo "$STEP. Generate keywords manually or install an AI backend"
fi
echo
STEP=$((STEP + 1))

echo "$STEP. Search your knowledge base:"
echo "   ./kbindex.py search \"your keywords\" --format json"
echo

if [ $READY -eq 1 ] && [ $HAS_KB_DIR -eq 1 ] && [ $HAS_DB -eq 1 ]; then
    echo -e "${GREEN}✓ Setup complete! You're ready to use kb-indexer.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Setup incomplete. Please follow the recommended actions above.${NC}"
    exit 1
fi
