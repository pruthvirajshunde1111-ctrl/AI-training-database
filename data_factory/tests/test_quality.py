"""Tests for the quality control modules."""

from data_factory.models import TaskExample, TaskType
from data_factory.quality.toxicity import ToxicityChecker
from data_factory.quality.bias import BiasDetector
from data_factory.quality.diversity import DiversityAnalyzer
from data_factory.quality.coherence import CoherenceEvaluator
from data_factory.quality.pipeline import QualityPipeline


def test_toxicity_clean():
    checker = ToxicityChecker(threshold=0.1)
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="What is the capital of France?",
        expected_output="Paris is the capital of France.",
    )
    report = checker.check(ex)
    assert report.scores["toxicity"] == 0.0
    assert report.passed is True


def test_toxicity_detected():
    checker = ToxicityChecker(threshold=0.1)
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="Why are you such an idiot?",
        expected_output="That is a stupid question.",
    )
    report = checker.check(ex)
    assert report.scores["toxicity"] > 0.0
    assert report.passed is False


def test_bias_detection():
    detector = BiasDetector(threshold=0.2)
    ex = TaskExample(
        task_type=TaskType.CLASSIFICATION,
        task_name="classification",
        input="All men are good at sports.",
        expected_output="stereotyping",
    )
    report = detector.check(ex)
    assert "bias" in report.scores


def test_diversity():
    analyzer = DiversityAnalyzer(threshold=0.3)
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="What is AI?",
        expected_output="AI is artificial intelligence.",
    )
    report = analyzer.check(ex)
    assert "diversity" in report.scores


def test_coherence():
    evaluator = CoherenceEvaluator(threshold=0.6)
    ex = TaskExample(
        task_type=TaskType.SUMMARIZATION,
        task_name="summarization",
        input="Long text passage here. It has multiple sentences. Each is complete.",
        expected_output="This is a coherent summary. It has proper structure.",
    )
    report = evaluator.check(ex)
    assert "coherence" in report.scores


def test_quality_pipeline():
    pipeline = QualityPipeline()
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="What is Python?",
        expected_output="Python is a programming language.",
    )
    report = pipeline.evaluate(ex)
    assert "toxicity" in report.scores
    assert "bias" in report.scores
    assert "diversity" in report.scores
    assert "coherence" in report.scores
    assert "overall" in report.scores
