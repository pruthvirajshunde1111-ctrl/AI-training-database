"""URL / web-page loader with fallback mechanisms.

Attempts to fetch via the Decodo API first (if configured), then
falls back to direct HTTP requests with Trafilatura for extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests

from data_factory.loaders.base import BaseLoader
from data_factory.models import Document, DocumentType


class URLLoader(BaseLoader):
    """Loads web pages and online resources as Document objects.

    Supports:
      - Direct HTTP fetch with content extraction (via trafilatura)
      - Decodo API integration for robust scraping
      - Configurable user-agent and timeout
    """

    def __init__(self, config: Optional = None) -> None:
        super().__init__(config)
        self._decodo_client = None
        if config and getattr(config, "decodo_api_key", None):
            try:
                from data_factory.integrations.decodo_api import DecodoClient

                self._decodo_client = DecodoClient(
                    api_key=config.decodo_api_key,
                    base_url=getattr(config, "decodo_base_url", None),
                )
                self.log.info("Decodo client initialized for URL loading")
            except Exception as e:
                self.log.warning("Failed to init Decodo client: %s", e)

    def _load_impl(self, source: str) -> List[Document]:
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        text = self._fetch_with_fallback(source)
        if not text.strip():
            raise RuntimeError(f"No content extracted from: {source}")

        return [
            Document(
                source=source,
                source_type=DocumentType.URL,
                content=text.strip(),
                metadata={
                    "url": source,
                    "domain": parsed.netloc,
                    "fetched_with": "decodo" if self._decodo_client else "trafilatura",
                },
            )
        ]

    def _fetch_with_fallback(self, url: str) -> str:
        if self._decodo_client:
            try:
                self.log.debug("Fetching via Decodo API: %s", url)
                return self._decodo_client.scrape(url)
            except Exception as e:
                self.log.warning("Decodo fetch failed, falling back: %s", e)

        return self._fetch_direct(url)

    def _fetch_direct(self, url: str) -> str:
        timeout = 30
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        if self.config:
            timeout = getattr(self.config, "request_timeout", 30)
            if hasattr(self.config, "user_agent"):
                headers["User-Agent"] = self.config.user_agent

        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        try:
            import trafilatura

            extracted = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if extracted:
                return extracted
        except ImportError:
            self.log.warning("trafilatura not installed; falling back to raw HTML")

        return resp.text

    def validate_source(self, source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
