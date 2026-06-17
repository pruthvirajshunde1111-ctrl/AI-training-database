"""Decodo API client — professional web scraping and data extraction.

Provides robust API integration with circuit-breaker fallback for
reliable content fetching.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from data_factory.utils.logger import LoggerMixin


class DecodoClient(LoggerMixin):
    """Client for the Decodo web scraping API.

    Usage::
        client = DecodoClient(api_key="...")
        html = client.scrape("https://example.com")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.decodo.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "DataFactory/0.1.0",
                    "Accept": "application/json",
                }
            )
        return self._session

    def scrape(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Scrape a URL using the Decodo API.

        Args:
            url: Target URL to scrape.
            options: Optional parameters (render_js, wait_for, etc.).

        Returns:
            Extracted text content.

        Raises:
            RuntimeError: If all retries or the API fails.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        payload: Dict[str, Any] = {"url": url, "format": "markdown"}
        if options:
            payload.update(options)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.log.debug(
                    "Decodo API call (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    url,
                )
                resp = self.session.post(
                    f"{self.base_url}/scrape",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                content = (
                    data.get("content")
                    or data.get("text")
                    or data.get("markdown")
                    or ""
                )
                if content:
                    return content

                raise RuntimeError(f"Empty content returned for {url}")

            except requests.Timeout as e:
                last_error = e
                self.log.warning(
                    "Decodo API timeout (attempt %d/%d)", attempt, self.max_retries
                )
            except requests.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response else 0
                if status in (401, 403):
                    raise RuntimeError(
                        "Decodo API authentication failed. Check your API key."
                    ) from e
                if status == 429:
                    import time

                    time.sleep(2**attempt)
                    continue
                if status >= 500:
                    continue
                raise
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Decodo API failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def health(self) -> Dict[str, Any]:
        """Check the Decodo API health status."""
        try:
            resp = self.session.get(
                f"{self.base_url}/health", timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
