"""Toxicity checker — inspects training examples for harmful content."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from data_factory.models import QualityReport, TaskExample
from data_factory.utils.logger import LoggerMixin

# Simple keyword-based toxicity detection (expand as needed)
TOXIC_PATTERNS: List[str] = [
    r"\b(hate|kill|murder|die|death)\b",
    r"\b(slut|whore|bitch|bastard)\b",
    r"\b(idiot|stupid|dumb)\b",
    r"\b(fucking|fuck)\b",
]

TOXIC_CONTEXT_PATTERNS: List[str] = [
    r"\b(hate speech|racial slurs|discriminat)\b",
    r"\b(violence|violent|abuse)\b",
]


class ToxicityChecker(LoggerMixin):
    """Evaluates toxicity levels in generated training examples.

    Uses keyword heuristics with option to integrate external APIs
    (e.g., Perspective API) for more accurate detection.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def check(self, example: TaskExample) -> QualityReport:
        """Check a single example for toxic content.

        Returns:
            :class:`QualityReport` with a ``toxicity`` score (0 = clean, 1 = toxic).
        """
        report = QualityReport(example_id=example.id)
        text = f"{example.input} {example.expected_output}".lower()

        score = self._compute_toxicity(text)
        report.add_score("toxicity", score)

        if score > self.threshold:
            report.add_issue(f"Toxicity score {score:.3f} exceeds threshold {self.threshold}")
            report.passed = False

        return report

    def check_batch(self, examples: List[TaskExample]) -> List[QualityReport]:
        return [self.check(ex) for ex in examples]

    def _compute_toxicity(self, text: str) -> float:
        severe = sum(1 for p in TOXIC_PATTERNS if re.search(p, text))
        contextual = sum(1 for p in TOXIC_CONTEXT_PATTERNS if re.search(p, text))

        score = (severe * 0.3 + contextual * 0.15)
        return min(round(score, 3), 1.0)

    @property
    def name(self) -> str:
        return "toxicity"
