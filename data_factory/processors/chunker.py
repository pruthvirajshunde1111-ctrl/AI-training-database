"""Text chunking module — segments cleaned text into processable pieces.

Supports multiple chunking strategies:
  - ``fixed``: Token-based fixed-size chunks with overlap
  - ``paragraph``: Paragraph-bounded chunks
  - ``sentence``: Sentence-bounded chunks
  - ``recursive``: Recursive split on separators (LangChain-style)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from data_factory.models import TextChunk
from data_factory.utils.logger import LoggerMixin


class TextChunker(LoggerMixin):
    """Splits cleaned text into :class:`TextChunk` objects."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
        min_chunk_length: int = 50,
        max_chunk_length: int = 2048,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

    def chunk(
        self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """Segment *text* into a list of :class:`TextChunk` objects."""
        metadata = metadata or {}
        strategy_map = {
            "fixed": self._chunk_fixed,
            "paragraph": self._chunk_paragraph,
            "sentence": self._chunk_sentence,
            "recursive": self._chunk_recursive,
        }

        splitter = strategy_map.get(self.strategy, self._chunk_recursive)
        segments = splitter(text)

        chunks: List[TextChunk] = []
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if len(seg) < self.min_chunk_length:
                continue
            if len(seg) > self.max_chunk_length:
                sub_segments = self._split_oversized(seg)
                for j, sub in enumerate(sub_segments):
                    chunk = TextChunk(
                        document_id=document_id,
                        index=len(chunks),
                        text=sub,
                        token_estimate=len(sub.split()),
                        metadata={**metadata, "sub_chunk": j},
                    )
                    chunks.append(chunk)
                continue

            chunk = TextChunk(
                document_id=document_id,
                index=len(chunks),
                text=seg,
                token_estimate=len(seg.split()),
                metadata=metadata,
            )
            chunks.append(chunk)

        self.log.info(
            "Chunked text into %d chunks (strategy=%s)", len(chunks), self.strategy
        )
        return chunks

    # ── chunking strategies ──────────────────────────────────────────

    def _chunk_fixed(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _chunk_paragraph(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_sentence(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_recursive(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, 0)

    def _recursive_split(
        self, text: str, separators: List[str], depth: int
    ) -> List[str]:
        if len(text) <= self.chunk_size or depth >= len(separators):
            return [text]

        sep = separators[depth]
        parts = text.split(sep)
        chunks: List[str] = []
        current: List[str] = []

        for part in parts:
            candidate = sep.join(current + [part])
            if len(candidate.split()) <= self.chunk_size:
                current.append(part)
            else:
                if current:
                    chunks.append(sep.join(current))
                current = [part]

        if current:
            chunks.append(sep.join(current))

        result: List[str] = []
        for chunk in chunks:
            if len(chunk.split()) > self.chunk_size:
                result.extend(self._recursive_split(chunk, separators, depth + 1))
            else:
                result.append(chunk)
        return result

    def _split_oversized(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        for i in range(0, len(words), self.max_chunk_length):
            chunks.append(" ".join(words[i : i + self.max_chunk_length]))
        return chunks
