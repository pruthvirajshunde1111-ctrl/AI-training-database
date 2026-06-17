"""Bias detector — identifies potential biases in training examples."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from data_factory.models import QualityReport, TaskExample
from data_factory.utils.logger import LoggerMixin

# Patterns that may indicate biased language
BIAS_PATTERNS: Dict[str, List[str]] = {
    "gender": [
        r"\b(men are|women are|boys are|girls are)\b",
        r"\b(all men|all women|always women|always men)\b",
    ],
    "racial": [
        r"\b(racial|racist|race-bait)\b",
    ],
    "stereotyping": [
        r"\b(typical (for|of) .+)\b",
        r"\b(always|never|every) (.+) (are|is)\b",
    ],
    "socioeconomic": [
        r"\b(poor people|rich people|upper class|lower class)\b",
    ],
}


class BiasDetector(LoggerMixin):
    """Detects potentially biased language in generated examples.

    Flags examples with high bias scores for review or filtering.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def check(self, example: TaskExample) -> QualityReport:
        """Check a single example for biased language.

        Returns:
            :class:`QualityReport` with a ``bias`` score (0 = unbiased, 1 = highly biased).
        """
        report = QualityReport(example_id=example.id)
        text = f"{example.input} {example.expected_output}".lower()

        score, details = self._compute_bias(text)
        report.add_score("bias", score)

        if score > self.threshold:
            for detail in details:
                report.add_issue(f"Bias detected: {detail} (score {score:.3f})")

        return report

    def check_batch(self, examples: List[TaskExample]) -> List[QualityReport]:
        return [self.check(ex) for ex in examples]

    def _compute_bias(self, text: str) -> Tuple[float, List[str]]:
        details: List[str] = []
        total_patterns = 0
        matches = 0

        for bias_type, patterns in BIAS_PATTERNS.items():
            for pattern in patterns:
                total_patterns += 1
                if re.search(pattern, text):
                    matches += 1
                    details.append(f"({bias_type}) matches pattern: {pattern}")

        score = matches / max(total_patterns, 1) if total_patterns > 0 else 0.0
        return min(round(score, 3), 1.0), details

    @property
    def name(self) -> str:
        return "bias"
