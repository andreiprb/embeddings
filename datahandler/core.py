"""
Main datahandler class for processing and managing text corpuses.
"""

import importlib
from pathlib import Path
from typing import Optional, Dict, Any

from .config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    FORCE_REPROCESS_FLAG
)
from .utils import (
    extract_text_from_docx,
    save_processed_text,
    load_processed_text,
    save_cache_metadata,
    is_cache_valid,
    clear_cache
)
from .processors.base import BaseProcessor


class DataHandler:
    """
    Main class for handling text corpus processing and caching.

    This class processes .docx corpus files on first run and caches the results
    for subsequent imports. It supports multiple languages and custom processors.
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE, force_reprocess: bool = FORCE_REPROCESS_FLAG):
        """
        Initialize datahandler for specified language.

        Args:
            language: Language code ('en' or 'ro')
            force_reprocess: If True, reprocess even if cache exists

        Raises:
            ValueError: If language is not supported
            FileNotFoundError: If corpus file doesn't exist
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language '{language}' not supported. Available: {list(SUPPORTED_LANGUAGES.keys())}")

        self.language = language
        self.config = SUPPORTED_LANGUAGES[language]
        self.force_reprocess = force_reprocess

        self.processed_text: Optional[str] = None
        self.processor: Optional[BaseProcessor] = None

        self._load_or_process()

    def _load_or_process(self) -> None:
        """
        Load processed text from cache or process from source.

        This method handles the core logic of checking cache validity
        and either loading from cache or processing from source.
        """
        source_file = self.config['source_file']
        cache_file = self.config['cache_file']

        use_cache = (
                not self.force_reprocess and
                is_cache_valid(self.language, source_file, cache_file)
        )

        if use_cache:
            print(f"Loading {self.config['display_name']} corpus from cache...")
            self.processed_text = load_processed_text(cache_file)
        else:
            print(f"Processing {self.config['display_name']} corpus from source...")
            self._process_from_source()

    def _process_from_source(self) -> None:
        """
        Process corpus from source .docx file.

        This method extracts text from the .docx file, processes it using
        the appropriate processor, and saves the result to cache.
        """
        source_file = self.config['source_file']
        cache_file = self.config['cache_file']

        raw_text = extract_text_from_docx(source_file)

        processor = self._get_processor()

        self.processed_text = processor.process_text(raw_text)

        save_processed_text(self.processed_text, cache_file)
        save_cache_metadata(self.language, source_file, cache_file)

        print(f"Processed and cached {len(self.processed_text)} characters.")

    def _get_processor(self) -> BaseProcessor:
        """
        Get the appropriate processor for the current language.

        Returns:
            BaseProcessor instance for the current language

        Raises:
            ImportError: If processor module cannot be imported
            AttributeError: If processor class is not found
        """
        if self.processor is None:
            try:
                # Map language codes to module names and class names
                processor_mapping = {
                    'en': ('english', 'EnglishProcessor'),
                    'ro': ('romanian', 'RomanianProcessor')
                }

                if self.language not in processor_mapping:
                    raise ValueError(f"No processor mapping for language '{self.language}'")

                module_name, class_name = processor_mapping[self.language]

                # Import the processor module dynamically
                full_module_name = f'.processors.{module_name}'
                module = importlib.import_module(full_module_name, package='datahandler')

                # Get the processor class
                processor_class = getattr(module, class_name)

                # Instantiate the processor
                self.processor = processor_class()

            except (ImportError, AttributeError) as e:
                raise ImportError(f"Could not load processor for language '{self.language}': {e}")

        return self.processor

    def get_text(self) -> str:
        """
        Get the processed text.

        Returns:
            Processed text content
        """
        if self.processed_text is None:
            raise RuntimeError("Text not loaded. This should not happen.")

        return self.processed_text

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current corpus and processing.

        Returns:
            Dictionary containing corpus information
        """
        return {
            'language': self.language,
            'display_name': self.config['display_name'],
            'source_file': str(self.config['source_file']),
            'cache_file': str(self.config['cache_file']),
            'text_length': len(self.processed_text) if self.processed_text else 0,
            'processor': self.processor.get_info() if self.processor else None
        }

    def reprocess(self) -> None:
        """
        Force reprocessing of the corpus.

        This method clears the cache and reprocesses the corpus from source.
        """
        print(f"Forcing reprocessing of {self.config['display_name']} corpus...")
        clear_cache(self.language)
        self._process_from_source()

    def clear_cache(self) -> None:
        """
        Clear the cache for this language.
        """
        clear_cache(self.language)
        print(f"Cache cleared for {self.config['display_name']} corpus.")

    @staticmethod
    def clear_all_cache() -> None:
        """
        Clear all cached data.
        """
        clear_cache()
        print("All cache cleared.")

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """
        Get dictionary of supported languages.

        Returns:
            Dictionary mapping language codes to display names
        """
        return {lang: config['display_name'] for lang, config in SUPPORTED_LANGUAGES.items()}

    def __repr__(self) -> str:
        """String representation of datahandler."""
        return f"datahandler(language='{self.language}', text_length={len(self.processed_text) if self.processed_text else 0})"

    def __len__(self) -> int:
        """Return length of processed text."""
        return len(self.processed_text) if self.processed_text else 0