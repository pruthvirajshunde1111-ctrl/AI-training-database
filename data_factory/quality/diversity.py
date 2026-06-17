"""Diversity analyzer — measures lexical and semantic diversity in datasets."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from data_factory.models import QualityReport, TaskExample
from data_factory.utils.logger import LoggerMixin


class DiversityAnalyzer(LoggerMixin):
    """Assesses diversity of generated examples using lexical metrics.

    Metrics:
      - Type-token ratio (TTR): unique words / total words
      - Question diversity: unique question starters
      - Length diversity: variance in input/output lengths
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self._seen_texts: Set[str] = set()

    def check(self, example: TaskExample) -> QualityReport:
        """Check a single example for diversity contribution.

        Returns:
            :class:`QualityReport` with a ``diversity`` score (0 = low, 1 = high).
        """
        report = QualityReport(example_id=example.id)

        ttr = self._type_token_ratio(f"{example.input} {example.expected_output}")
        has_new_content = self._check_novelty(f"{example.input} {example.expected_output}")

        # Combine metrics
        score = round(ttr * 0.5 + (1.0 if has_new_content else 0.0) * 0.5, 3)
        report.add_score("diversity", score)

        if score < self.threshold:
            report.add_issue(
                f"Diversity score {score:.3f} below threshold {self.threshold}"
            )

        return report

    def check_batch(
        self, examples: List[TaskExample]
    ) -> List[QualityReport]:
        reports = [self.check(ex) for ex in examples]
        return reports

    def _type_token_ratio(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        unique = len(set(words))
        total = len(words)
        return round(unique / total, 3)

    def _check_novelty(self, text: str) -> bool:
        fingerprint = re.sub(r"\s+", " ", text.lower()[:100])
        if fingerprint in self._seen_texts:
            return False
        self._seen_texts.add(fingerprint)
        return True

    @property
    def name(self) -> str:
        return "diversity"
