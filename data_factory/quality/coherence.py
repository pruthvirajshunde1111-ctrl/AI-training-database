"""Coherence evaluator — checks logical consistency and readability."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from data_factory.models import QualityReport, TaskExample
from data_factory.utils.logger import LoggerMixin


class CoherenceEvaluator(LoggerMixin):
    """Evaluates coherence of generated training examples.

    Metrics:
      - Readability score (based on sentence length / complexity)
      - Input-output alignment (keyword overlap)
      - Structural completeness (proper punctuation / sentence boundaries)
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def check(self, example: TaskExample) -> QualityReport:
        """Check a single example for coherence.

        Returns:
            :class:`QualityReport` with a ``coherence`` score (0 = low, 1 = high).
        """
        report = QualityReport(example_id=example.id)

        readability = self._readability(example.expected_output)
        alignment = self._alignment(example.input, example.expected_output)
        completeness = self._completeness(example.expected_output)

        score = round(readability * 0.3 + alignment * 0.4 + completeness * 0.3, 3)
        report.add_score("coherence", score)

        if score < self.threshold:
            report.add_issue(
                f"Coherence score {score:.3f} below threshold {self.threshold}"
            )

        return report

    def check_batch(self, examples: List[TaskExample]) -> List[QualityReport]:
        return [self.check(ex) for ex in examples]

    def _readability(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        avg_sentence_len = len(words) / len(sentences)
        score = 1.0 / (1.0 + avg_sentence_len / 20.0)
        return round(score, 3)

    def _alignment(self, inp: str, out: str) -> float:
        inp_words = set(inp.lower().split())
        out_words = set(out.lower().split())
        if not inp_words or not out_words:
            return 0.0

        intersection = inp_words & out_words
        overlap = len(intersection) / max(len(inp_words), len(out_words))
        return round(overlap, 3)

    def _completeness(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        has_capitals = sum(1 for s in sentences if s and s[0].isupper())
        has_endings = sum(
            1 for s in sentences if s and s[-1] in ".!?"
        )

        ratio = (has_capitals + has_endings) / (len(sentences) * 2)
        return round(ratio, 3)

    @property
    def name(self) -> str:
        return "coherence"
