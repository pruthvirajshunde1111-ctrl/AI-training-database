"""PDF document loader using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from data_factory.loaders.base import BaseLoader
from data_factory.models import Document, DocumentType


class PDFLoader(BaseLoader):
    """Loads text content from PDF files using PyMuPDF."""

    def _load_impl(self, source: str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        import fitz

        docs: List[Document] = []
        with fitz.open(path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if not text:
                    continue
                doc = Document(
                    source=str(path),
                    source_type=DocumentType.PDF,
                    content=text,
                    metadata={
                        "page": page_num,
                        "total_pages": len(pdf),
                        "file_size_bytes": path.stat().st_size,
                    },
                )
                docs.append(doc)

        return docs

    def validate_source(self, source: str) -> bool:
        path = Path(source)
        if not path.exists() or path.suffix.lower() != ".pdf":
            return False
        # Quick magic-byte check
        try:
            header = path.read_bytes()[:5]
            return header == b"%PDF-"
        except Exception:
            return False
