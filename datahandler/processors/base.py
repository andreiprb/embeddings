"""
Base processor interface for text processing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseProcessor(ABC):
    """
    Abstract base class for text processors.

    Each language-specific processor should inherit from this class
    and implement the process_text method.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor.

        Args:
            config: Optional configuration dictionary for the processor
        """
        self.config = config or {}

    @abstractmethod
    def process_text(self, raw_text: str) -> str:
        """
        Process raw text and return processed text.

        Args:
            raw_text: Raw text extracted from the corpus

        Returns:
            Processed text ready for use
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Common preprocessing steps that can be overridden.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text

    def postprocess(self, text: str) -> str:
        """
        Common postprocessing steps that can be overridden.

        Args:
            text: Processed text

        Returns:
            Final processed text
        """
        return text.strip()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this processor.

        Returns:
            Dictionary containing processor information
        """
        return {
            'name': self.__class__.__name__,
            'config': self.config
        }