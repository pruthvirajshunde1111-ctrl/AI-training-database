"""Abstract base class for all document loaders.

Provides a consistent interface and shared utilities so that adding a
new data source requires nothing more than subclassing and implementing
``_load_impl()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_factory.models import Document, DocumentType
from data_factory.utils.logger import LoggerMixin


class BaseLoader(ABC, LoggerMixin):
    """Every document loader in the factory inherits from this class.

    Subclasses must implement :meth:`_load_impl` which returns one or
    more :class:`Document` objects.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._source_count: int = 0

    @abstractmethod
    def _load_impl(self, source: str) -> List[Document]:
        """Perform the actual loading logic for a single source."""

    def load(self, source: str) -> List[Document]:
        """Load a document from the given source.

        Args:
            source: File path, URL, or other identifier.

        Returns:
            List of :class:`Document` instances.
        """
        self.log.info("Loading source: %s", source)
        docs = self._load_impl(source)
        for doc in docs:
            doc.compute_word_count()
            doc.metadata["loader"] = self.__class__.__name__
        self._source_count += 1
        self.log.info("Loaded %d document(s) from: %s", len(docs), source)
        return docs

    def load_batch(self, sources: List[str]) -> List[Document]:
        """Load multiple sources sequentially.

        Args:
            sources: List of file paths, URLs, etc.

        Returns:
            Flat list of all :class:`Document` objects.
        """
        all_docs: List[Document] = []
        for src in sources:
            try:
                all_docs.extend(self.load(src))
            except Exception as e:
                self.log.error("Failed to load '%s': %s", src, e)
        return all_docs

    @property
    def source_count(self) -> int:
        return self._source_count

    def validate_source(self, source: str) -> bool:
        """Check whether a source exists / is reachable (override in subclasses)."""
        path = Path(source)
        return path.exists()

    @staticmethod
    def detect_type(source: str) -> DocumentType:
        path = Path(source)
        suffix = path.suffix.lower()
        mapping: Dict[str, DocumentType] = {
            ".pdf": DocumentType.PDF,
            ".txt": DocumentType.TEXT,
            ".md": DocumentType.MARKDOWN,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
            ".csv": DocumentType.CSV,
            ".json": DocumentType.JSON,
        }
        return mapping.get(suffix, DocumentType.UNKNOWN)
