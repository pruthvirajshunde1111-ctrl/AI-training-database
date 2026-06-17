"""Text cleaning module — the first station in the processing pipeline.

Removes noise, normalises content, and prepares raw text for chunking.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from data_factory.utils.logger import LoggerMixin


class TextCleaner(LoggerMixin):
    """Configurable text cleaner that applies a series of cleaning steps.

    Built-in steps (configured via ``cleaning_steps``):
      - ``strip_whitespace`` — Collapse repeated whitespace
      - ``normalize_unicode`` — NFC normalise & remove non-printable chars
      - ``remove_urls`` — Strip http(s) URLs
      - ``remove_html`` — Strip HTML tags
      - ``remove_emails`` — Remove email addresses
      - ``remove_numbers`` — Remove standalone numbers (optional)
      - ``to_lowercase`` — Lowercase text (use with caution)
    """

    STEPS: Dict[str, Callable[[str], str]] = {}

    def __init__(self, config: Optional = None) -> None:
        self.config = config
        self._steps: List[str] = [
            "strip_whitespace",
            "normalize_unicode",
            "remove_urls",
            "remove_html",
        ]
        if config and hasattr(config, "cleaning_steps"):
            self._steps = config.cleaning_steps

    @classmethod
    def _register(cls) -> None:
        cls.STEPS = {
            "strip_whitespace": cls._strip_whitespace,
            "normalize_unicode": cls._normalize_unicode,
            "remove_urls": cls._remove_urls,
            "remove_html": cls._remove_html,
            "remove_emails": cls._remove_emails,
            "remove_numbers": cls._remove_numbers,
            "to_lowercase": cls._to_lowercase,
        }

    def clean(self, text: str) -> str:
        """Apply all configured cleaning steps to *text*."""
        if not self.STEPS:
            self._register()

        for name in self._steps:
            fn = self.STEPS.get(name)
            if fn:
                text = fn(text)
        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    # ── built-in cleaning implementations ────────────────────────────

    @staticmethod
    def _strip_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        import unicodedata

        text = unicodedata.normalize("NFC", text)
        return "".join(ch if ch == "\n" or ch.isprintable() else " " for ch in text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"https?://\S+", "", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _remove_emails(text: str) -> str:
        return re.sub(r"\S+@\S+\.\S+", "", text)

    @staticmethod
    def _remove_numbers(text: str) -> str:
        return re.sub(r"\b\d+\b", "", text)

    @staticmethod
    def _to_lowercase(text: str) -> str:
        return text.lower()
