"""Local file loader supporting plain text, Markdown, JSON, and CSV."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_factory.loaders.base import BaseLoader
from data_factory.models import Document, DocumentType


class FileLoader(BaseLoader):
    """Loads supported local files and returns Document objects.

    Supported formats: .txt, .md, .json, .csv, .html
    """

    ENCODING = "utf-8"

    def _load_impl(self, source: str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        doc_type = self.detect_type(source)
        raw = path.read_text(encoding=self.ENCODING)

        if doc_type == DocumentType.CSV:
            return self._load_csv(path, raw)
        if doc_type == DocumentType.JSON:
            return self._load_json(path, raw)

        return [
            Document(
                source=str(path),
                source_type=doc_type,
                content=raw.strip(),
                metadata={
                    "file_name": path.name,
                    "file_size_bytes": path.stat().st_size,
                    "extension": path.suffix,
                },
            )
        ]

    def _load_csv(self, path: Path, raw: str) -> List[Document]:
        docs: List[Document] = []
        reader = csv.DictReader(StringIO(raw))
        rows = list(reader)
        if not rows:
            return docs
        headers = list(rows[0].keys())

        for i, row in enumerate(rows):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            doc = Document(
                source=str(path),
                source_type=DocumentType.CSV,
                content=text,
                metadata={
                    "row": i,
                    "total_rows": len(rows),
                    "headers": headers,
                    "file_name": path.name,
                },
            )
            docs.append(doc)
        return docs

    def _load_json(self, path: Path, raw: str) -> List[Document]:
        docs: List[Document] = []
        data = json.loads(raw)

        if isinstance(data, list):
            for i, item in enumerate(data):
                text = json.dumps(item, ensure_ascii=False, indent=2)
                doc = Document(
                    source=str(path),
                    source_type=DocumentType.JSON,
                    content=text,
                    metadata={
                        "index": i,
                        "total_items": len(data),
                        "file_name": path.name,
                    },
                )
                docs.append(doc)
        elif isinstance(data, dict):
            text = json.dumps(data, ensure_ascii=False, indent=2)
            docs.append(
                Document(
                    source=str(path),
                    source_type=DocumentType.JSON,
                    content=text,
                    metadata={"file_name": path.name},
                )
            )
        return docs

    def validate_source(self, source: str) -> bool:
        path = Path(source)
        return path.exists() and path.is_file()
