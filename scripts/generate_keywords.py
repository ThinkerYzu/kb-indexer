#!/usr/bin/env python3
"""
Generate keywords.json file for a markdown document using LLM.

Uses Ollama by default (local, free) with fallback to Gemini (cloud, API key required).
"""

import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Try to import LLM backends
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from google import genai
    from dotenv import load_dotenv
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def read_markdown(filepath: str) -> str:
    """Read markdown file content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def generate_keywords_with_ollama(content: str, model: str = "llama3.2:3b") -> Dict:
    """Generate keywords using Ollama."""
    # Limit content length to avoid context issues (first 8000 chars should be enough)
    content_preview = content[:8000] if len(content) > 8000 else content
    if len(content) > 8000:
        content_preview += "\n\n[... document truncated for analysis ...]"

    prompt = f"""You are a metadata extraction assistant. Your task is to analyze a markdown document and extract keywords and metadata.

TASK: Generate a JSON object with this exact structure:
{{
  "filepath": "filename.md",
  "title": "Document title (extract from content or create descriptive title)",
  "summary": "2-3 sentence summary of the document's main points",
  "keywords": ["keyword1", "keyword2", ...],
  "categories": {{
    "primary": ["main topics"],
    "concepts": ["key concepts and ideas"],
    "tools": ["tools, libraries, frameworks mentioned"],
    "abbreviations": ["abbreviations and acronyms"]
  }}
}}

GUIDELINES:
- Extract 10-30 keywords covering main topics, concepts, tools, technologies
- Include both specific terms and general concepts
- Include abbreviations separately in the abbreviations category
- Make the summary concise but informative
- Use lowercase for keywords unless they are proper nouns or abbreviations
- The title should be clear, descriptive, and in Title Case (capitalize major words)
- If the document has a title header (# Title), use it; otherwise create one

IMPORTANT: Return ONLY a valid JSON object. Do not include any code, explanations, or other text.

DOCUMENT TO ANALYZE:
{content_preview}

JSON OUTPUT:"""

    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    # Parse the response
    response_text = response['message']['content'].strip()

    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        # Remove first line (```) and last line (```)
        response_text = '\n'.join(lines[1:-1])
        # Remove language identifier if present (e.g., ```json)
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        # Debug: print the response to stderr
        print(f"DEBUG: LLM response length: {len(response_text)}", file=sys.stderr)
        print(f"DEBUG: First 500 chars: {response_text[:500]}", file=sys.stderr)
        print(f"DEBUG: Last 500 chars: {response_text[-500:]}", file=sys.stderr)
        raise


def generate_keywords_with_gemini(content: str, model: str = "gemini-2.0-flash-exp") -> Dict:
    """Generate keywords using Gemini."""
    load_dotenv()
    client = genai.Client()

    # Limit content length to avoid context issues (first 8000 chars should be enough)
    content_preview = content[:8000] if len(content) > 8000 else content
    if len(content) > 8000:
        content_preview += "\n\n[... document truncated for analysis ...]"

    prompt = f"""You are a metadata extraction assistant. Your task is to analyze a markdown document and extract keywords and metadata.

TASK: Generate a JSON object with this exact structure:
{{
  "filepath": "filename.md",
  "title": "Document title (extract from content or create descriptive title)",
  "summary": "2-3 sentence summary of the document's main points",
  "keywords": ["keyword1", "keyword2", ...],
  "categories": {{
    "primary": ["main topics"],
    "concepts": ["key concepts and ideas"],
    "tools": ["tools, libraries, frameworks mentioned"],
    "abbreviations": ["abbreviations and acronyms"]
  }}
}}

GUIDELINES:
- Extract 10-30 keywords covering main topics, concepts, tools, technologies
- Include both specific terms and general concepts
- Include abbreviations separately in the abbreviations category
- Make the summary concise but informative
- Use lowercase for keywords unless they are proper nouns or abbreviations
- The title should be clear, descriptive, and in Title Case (capitalize major words)
- If the document has a title header (# Title), use it; otherwise create one

IMPORTANT: Return ONLY a valid JSON object. Do not include any code, explanations, or other text.

DOCUMENT TO ANALYZE:
{content_preview}

JSON OUTPUT:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    response_text = response.text.strip()

    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()

    return json.loads(response_text)


def generate_keywords_with_claude(content: str, md_filepath: str) -> Dict:
    """Generate keywords using Claude Code CLI."""
    basename = os.path.basename(md_filepath)

    prompt = f"""Extract keywords and metadata from this markdown document and return ONLY a valid JSON object with this structure:

{{
  "filepath": "{basename}",
  "title": "Document title (extract from content or create descriptive title in Title Case)",
  "summary": "2-3 sentence summary of the document's main points",
  "keywords": ["keyword1", "keyword2", ...],
  "categories": {{
    "primary": ["main topics"],
    "concepts": ["key concepts and ideas"],
    "tools": ["tools, libraries, frameworks mentioned"],
    "abbreviations": ["abbreviations and acronyms"]
  }}
}}

Guidelines:
- Extract 10-30 keywords covering main topics, concepts, tools, technologies
- Include both specific terms and general concepts
- Use lowercase for keywords unless they are proper nouns or abbreviations
- If the document has a title header (# Title), use it; otherwise create one

Return ONLY the JSON object, no other text or markdown formatting.

Document content:
{content}"""

    # Run claude command with prompt via stdin
    # Restrict tools to only Read (no file modifications, no bash commands)
    result = subprocess.run(
        ['claude', '-p', '--allowed-tools', 'Read'],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude command failed: {result.stderr}")

    response_text = result.stdout.strip()

    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1])
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()

    return json.loads(response_text)


def generate_keywords(md_filepath: str, backend: str = "auto", model: Optional[str] = None) -> Dict:
    """
    Generate keywords for a markdown document.

    Args:
        md_filepath: Path to markdown file
        backend: "auto", "claude", "ollama", or "gemini"
        model: Optional model name override

    Returns:
        Dict with keywords metadata
    """
    content = read_markdown(md_filepath)
    basename = os.path.basename(md_filepath)

    # Auto-select backend
    if backend == "auto":
        # Prefer Claude Code if available
        if subprocess.run(['which', 'claude'], capture_output=True).returncode == 0:
            backend = "claude"
        elif OLLAMA_AVAILABLE:
            backend = "ollama"
        elif GEMINI_AVAILABLE:
            backend = "gemini"
        else:
            raise RuntimeError("No LLM backend available. Install claude, ollama, or google-genai.")

    # Generate keywords
    if backend == "claude":
        keywords_data = generate_keywords_with_claude(content, md_filepath)
    elif backend == "ollama":
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install: pip install ollama")
        keywords_data = generate_keywords_with_ollama(
            content,
            model or "llama3.2:3b"
        )
    elif backend == "gemini":
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini not available. Install: pip install google-genai python-dotenv")
        keywords_data = generate_keywords_with_gemini(
            content,
            model or "gemini-2.0-flash-exp"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Override filepath with actual filename
    keywords_data["filepath"] = basename

    return keywords_data


def save_keywords(keywords_data: Dict, output_filepath: str):
    """Save keywords data to JSON file."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(keywords_data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: generate_keywords.py <markdown_file> [--backend ollama|gemini] [--model <model_name>]", file=sys.stderr)
        sys.exit(1)

    md_filepath = sys.argv[1]

    # Parse optional arguments
    backend = "auto"
    model = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if not os.path.exists(md_filepath):
        print(f"Error: File not found: {md_filepath}", file=sys.stderr)
        sys.exit(1)

    # Generate output filepath
    output_filepath = md_filepath.rsplit('.md', 1)[0] + '.keywords.json'

    try:
        print(f"Generating keywords for: {md_filepath}", file=sys.stderr)
        print(f"Using backend: {backend}", file=sys.stderr)

        keywords_data = generate_keywords(md_filepath, backend, model)
        save_keywords(keywords_data, output_filepath)

        print(f"✓ Generated: {output_filepath}", file=sys.stderr)
        print(output_filepath)  # Output filepath to stdout for script use

    except Exception as e:
        print(f"Error generating keywords: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
