"""Tests for the core data models."""

import json
from data_factory.models import Document, TextChunk, TaskExample, TaskTemplate, Dataset, TaskType, QualityReport, QualityMetric


def test_document_creation():
    doc = Document(source="test.txt", source_type="text", content="Hello world")
    assert doc.id.startswith("doc_")
    assert doc.content == "Hello world"
    assert doc.word_count is None
    doc.compute_word_count()
    assert doc.word_count == 2


def test_document_empty_content():
    import pytest
    with pytest.raises(ValueError):
        Document(source="test.txt", source_type="text", content="")


def test_text_chunk():
    chunk = TextChunk(document_id="doc_abc", index=0, text="Some text here")
    assert chunk.id.startswith("chunk_")
    assert chunk.token_estimate == 0
    chunk.compute_token_estimate()
    assert chunk.token_estimate == 3


def test_task_example():
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa_generation",
        input="What is AI?",
        expected_output="Artificial Intelligence",
    )
    assert ex.id.startswith("ex_")
    d = ex.to_dict()
    assert d["task_type"] == "qa"
    assert d["input"] == "What is AI?"


def test_task_template():
    tmpl = TaskTemplate(
        name="test",
        task_type=TaskType.CLASSIFICATION,
        description="Test template",
        system_prompt="You are a classifier",
        user_prompt_template="Classify: {text}",
    )
    assert tmpl.id.startswith("tmpl_")
    assert tmpl.task_type == TaskType.CLASSIFICATION


def test_quality_report():
    report = QualityReport(example_id="ex_123")
    assert report.passed is True
    report.add_score(QualityMetric.TOXICITY, 0.05)
    report.add_score(QualityMetric.COHERENCE, 0.85)
    assert report.scores["toxicity"] == 0.05
    assert report.scores["coherence"] == 0.85
    assert report.overall_score == pytest.approx(0.45)


def test_dataset():
    ds = Dataset(name="test_dataset")
    assert ds.size == 0
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="Q?",
        expected_output="A",
    )
    ds.add_example(ex)
    assert ds.size == 1
    assert ds.examples[0].id == ex.id


def test_dataset_export_jsonl(tmp_path):
    ds = Dataset(name="export_test")
    ex = TaskExample(
        task_type=TaskType.QA,
        task_name="qa",
        input="Q?",
        expected_output="A",
    )
    ds.add_example(ex)
    path = tmp_path / "test.jsonl"
    result = ds.export(str(path), format="jsonl")
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["input"] == "Q?"


import pytest
