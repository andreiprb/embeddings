"""
Utility functions for datahandler package.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import mammoth
from datetime import datetime

from .config import CACHE_DIR, CACHE_METADATA_FILE, VALIDATE_CACHE_TIMESTAMPS


def ensure_cache_directory():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def extract_text_from_docx(docx_path: Path) -> str:
    """
    Extract plain text from a .docx file using mammoth.

    Args:
        docx_path: Path to the .docx file

    Returns:
        Extracted plain text

    Raises:
        FileNotFoundError: If the docx file doesn't exist
        Exception: If text extraction fails
    """
    if not docx_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {docx_path}")

    try:
        with open(docx_path, "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value
    except Exception as e:
        raise Exception(f"Failed to extract text from {docx_path}: {str(e)}")


def save_processed_text(text: str, cache_file: Path) -> None:
    """
    Save processed text to cache file.

    Args:
        text: Processed text to save
        cache_file: Path where to save the text
    """
    ensure_cache_directory()
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)


def load_processed_text(cache_file: Path) -> str:
    """
    Load processed text from cache file.

    Args:
        cache_file: Path to the cache file

    Returns:
        Loaded text content

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    with open(cache_file, 'r', encoding='utf-8') as f:
        return f.read()


def save_cache_metadata(language: str, source_file: Path, cache_file: Path) -> None:
    """
    Save metadata about the cached file.

    Args:
        language: Language code
        source_file: Path to source .docx file
        cache_file: Path to cache file
    """
    ensure_cache_directory()

    metadata = {}
    if CACHE_METADATA_FILE.exists():
        with open(CACHE_METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    metadata[language] = {
        'source_file': str(source_file),
        'cache_file': str(cache_file),
        'processed_at': datetime.now().isoformat(),
        'source_modified': datetime.fromtimestamp(source_file.stat().st_mtime).isoformat()
    }

    with open(CACHE_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def load_cache_metadata() -> Dict[str, Any]:
    """
    Load cache metadata.

    Returns:
        Dictionary containing cache metadata
    """
    if not CACHE_METADATA_FILE.exists():
        return {}

    with open(CACHE_METADATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_cache_valid(language: str, source_file: Path, cache_file: Path) -> bool:
    """
    Check if cache is valid for the given language.

    Args:
        language: Language code
        source_file: Path to source .docx file
        cache_file: Path to cache file

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_file.exists():
        return False

    if not VALIDATE_CACHE_TIMESTAMPS:
        return True

    metadata = load_cache_metadata()
    if language not in metadata:
        return False

    try:
        source_modified = datetime.fromtimestamp(source_file.stat().st_mtime)
        cache_source_modified = datetime.fromisoformat(metadata[language]['source_modified'])

        return source_modified <= cache_source_modified
    except (KeyError, OSError, ValueError):
        return False


def clear_cache(language: Optional[str] = None) -> None:
    """
    Clear cache for specified language or all languages.

    Args:
        language: Language code to clear, or None to clear all
    """
    if language:
        metadata = load_cache_metadata()
        if language in metadata:
            cache_file = Path(metadata[language]['cache_file'])
            if cache_file.exists():
                cache_file.unlink()
            del metadata[language]

            with open(CACHE_METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    else:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        ensure_cache_directory()