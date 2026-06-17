"""Quality Pipeline — orchestrates all quality checks into one flow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_factory.models import QualityReport, TaskExample
from data_factory.quality.toxicity import ToxicityChecker
from data_factory.quality.bias import BiasDetector
from data_factory.quality.diversity import DiversityAnalyzer
from data_factory.quality.coherence import CoherenceEvaluator
from data_factory.utils.logger import LoggerMixin


class QualityPipeline(LoggerMixin):
    """Combines all inspection stations into a single quality gate.

    Runs toxicity, bias, diversity, and coherence checks on each
    training example, producing a comprehensive quality report.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

        self.toxicity = ToxicityChecker(
            threshold=self._get_threshold("toxicity", 0.1)
        )
        self.bias = BiasDetector(
            threshold=self._get_threshold("bias", 0.2)
        )
        self.diversity = DiversityAnalyzer(
            threshold=self._get_threshold("diversity", 0.3)
        )
        self.coherence = CoherenceEvaluator(
            threshold=self._get_threshold("coherence", 0.6)
        )

        self._checkers = [
            self.toxicity,
            self.bias,
            self.diversity,
            self.coherence,
        ]

    def _get_threshold(self, name: str, default: float) -> float:
        if self.config and hasattr(self.config, "quality_thresholds"):
            return self.config.quality_thresholds.get(name, default)
        return default

    def evaluate(self, example: TaskExample) -> QualityReport:
        """Run all quality checks on a single example."""
        report = QualityReport(example_id=example.id)
        passed_all = True

        for checker in self._checkers:
            checker_report = checker.check(example)
            for metric, score in checker_report.scores.items():
                report.add_score(metric, score)
            report.issues.extend(checker_report.issues)
            if not checker_report.passed:
                passed_all = False

        report.passed = passed_all
        report.add_score("overall", report.overall_score)

        if not passed_all:
            scores_str = ", ".join(f"{k}={v:.3f}" for k, v in report.scores.items())
            self.log.debug(
                "Example %s FAILED QC: scores=[%s] issues=[%s]",
                example.id[:8],
                scores_str,
                "; ".join(report.issues),
            )

        return report

    def evaluate_batch(
        self, examples: List[TaskExample]
    ) -> List[QualityReport]:
        """Run all quality checks on a batch of examples."""
        self.log.info(
            "Running quality pipeline on %d examples...", len(examples)
        )
        reports = []

        for ex in examples:
            try:
                report = self.evaluate(ex)
                reports.append(report)
            except Exception as e:
                self.log.error(
                    "Quality check failed for example %s: %s", ex.id, e
                )
                report = QualityReport(
                    example_id=ex.id,
                    scores={"overall": 0.0},
                    issues=[str(e)],
                    passed=False,
                )
                reports.append(report)

        passed = sum(1 for r in reports if r.passed)
        self.log.info(
            "Quality pipeline complete: %d/%d passed", passed, len(reports)
        )
        return reports

    def filter(
        self, examples: List[TaskExample]
    ) -> List[TaskExample]:
        """Filter examples, keeping only those that pass all quality checks."""
        reports = self.evaluate_batch(examples)
        passed_ids = {r.example_id for r in reports if r.passed}
        return [ex for ex in examples if ex.id in passed_ids]
