"""Web scraper module with multiple fallback strategies.

Provides a unified scraping interface that tries multiple backends:
  1. Decodo API (if configured)
  2. Trafilatura (best-in-class extraction)
  3. BeautifulSoup (manual extraction)
  4. Raw requests (last resort)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from data_factory.utils.logger import LoggerMixin


class WebScraper(LoggerMixin):
    """Scrapes web content with automatic fallback between backends."""

    FALLBACK_CHAIN: List[str] = ["decodo", "trafilatura", "beautifulsoup", "raw"]

    def __init__(
        self,
        config: Optional[Any] = None,
        decodo_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.decodo_client = decodo_client
        self.timeout = 30
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        if config:
            self.timeout = getattr(config, "request_timeout", 30)
            self.user_agent = getattr(config, "user_agent", self.user_agent)

    def scrape(
        self, url: str, backends: Optional[List[str]] = None
    ) -> str:
        """Scrape *url* with automatic fallback between backends.

        Args:
            url: Target URL.
            backends: Ordered list of backends to try (default: full chain).

        Returns:
            Extracted text content.

        Raises:
            RuntimeError: If all backends fail.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL: {url}")

        backends = backends or self.FALLBACK_CHAIN
        last_error: Optional[Exception] = None

        for backend in backends:
            try:
                self.log.debug("Scraping with backend: %s", backend)
                return self._call_backend(backend, url)
            except Exception as e:
                last_error = e
                self.log.warning("Backend '%s' failed: %s", backend, e)
                continue

        raise RuntimeError(
            f"All scraping backends failed for {url}: {last_error}"
        )

    def _call_backend(self, backend: str, url: str) -> str:
        backend_map = {
            "decodo": self._scrape_decodo,
            "trafilatura": self._scrape_trafilatura,
            "beautifulsoup": self._scrape_beautifulsoup,
            "raw": self._scrape_raw,
        }

        fn = backend_map.get(backend)
        if fn is None:
            raise ValueError(f"Unknown backend: {backend}")
        return fn(url)

    def _scrape_decodo(self, url: str) -> str:
        if self.decodo_client is None:
            raise RuntimeError("Decodo client not configured")
        return self.decodo_client.scrape(url)

    def _scrape_trafilatura(self, url: str) -> str:
        try:
            import trafilatura
        except ImportError:
            raise RuntimeError("trafilatura not installed")

        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise RuntimeError("trafilatura: failed to fetch URL")

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted is None:
            raise RuntimeError("trafilatura: no content extracted")
        return extracted

    def _scrape_beautifulsoup(self, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise RuntimeError("beautifulsoup4 not installed")

        resp = requests.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _scrape_raw(self, url: str) -> str:
        resp = requests.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        text = resp.text
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def scrape_batch(self, urls: List[str]) -> Dict[str, str]:
        """Scrape multiple URLs, returning a mapping of URL to content."""
        results: Dict[str, str] = {}
        for url in urls:
            try:
                results[url] = self.scrape(url)
            except Exception as e:
                self.log.error("Failed to scrape %s: %s", url, e)
                results[url] = ""
        return results
