"""
datahandler package for processing text corpuses.

This package provides a datahandler class that processes .docx corpus files
and caches the results for fast subsequent access.

Usage:
    from datahandler import datahandler

    # Default English corpus
    handler = datahandler()
    text = handler.get_text()

    # Romanian corpus
    handler_ro = datahandler('ro')
    text_ro = handler_ro.get_text()
"""

from .core import DataHandler
from .config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE

__version__ = "1.0.0"
__author__ = "Andrei Priboi"
__email__ = "andreipriboi@icloud.com"

__all__ = ['DataHandler', 'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE']

print(f"datahandler v{__version__} initialized. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")