"""Processing pipeline — orchestrates cleaning and chunking into one flow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_factory.models import Document, TextChunk
from data_factory.processors.cleaner import TextCleaner
from data_factory.processors.chunker import TextChunker
from data_factory.utils.logger import LoggerMixin


class ProcessingPipeline(LoggerMixin):
    """Combines cleaning and chunking for a complete document-to-chunks flow.

    Usage::
        pipeline = ProcessingPipeline(config)
        chunks = pipeline.process(document)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

        chunk_size = getattr(config, "chunk_size", 512) if config else 512
        chunk_overlap = getattr(config, "chunk_overlap", 64) if config else 64
        min_chunk_length = getattr(config, "min_chunk_length", 50) if config else 50
        max_chunk_length = getattr(config, "max_chunk_length", 2048) if config else 2048

        self.cleaner = TextCleaner(config)
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="recursive",
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )

    def process(self, document: Document) -> List[TextChunk]:
        """Clean and chunk a single document."""
        self.log.info("Processing document: %s", document.id)

        cleaned = self.cleaner.clean(document.content)
        if not cleaned:
            self.log.warning("Document %s produced empty text after cleaning", document.id)
            return []

        chunks = self.chunker.chunk(
            text=cleaned,
            document_id=document.id,
            metadata={
                "source": document.source,
                "source_type": document.source_type.value,
            },
        )

        return chunks

    def process_batch(self, documents: List[Document]) -> List[TextChunk]:
        """Process multiple documents into a flat list of chunks."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            try:
                all_chunks.extend(self.process(doc))
            except Exception as e:
                self.log.error("Failed to process document %s: %s", doc.id, e)
        return all_chunks
