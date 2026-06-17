"""Tests for the text processing pipeline."""

from data_factory.processors.cleaner import TextCleaner
from data_factory.processors.chunker import TextChunker
from data_factory.processors.pipeline import ProcessingPipeline
from data_factory.models import Document


def test_cleaner_strip_whitespace():
    cleaner = TextCleaner()
    result = cleaner.clean("  Hello    world  ")
    assert result == "Hello world"


def test_cleaner_remove_urls():
    cleaner = TextCleaner()
    result = cleaner.clean("Check https://example.com/page for details")
    assert "https://" not in result


def test_cleaner_remove_html():
    cleaner = TextCleaner()
    result = cleaner.clean("<p>Hello <b>world</b></p>")
    assert "<p>" not in result
    assert "Hello world" in result


def test_chunker_fixed():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2, strategy="fixed", min_chunk_length=1)
    text = "one two three four five six seven eight nine ten eleven twelve"
    chunks = chunker.chunk(text, document_id="doc_1")
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.text.split()) <= 12  # max_chunk_length


def test_chunker_paragraph():
    chunker = TextChunker(strategy="paragraph", min_chunk_length=1)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunker.chunk(text, document_id="doc_1")
    assert len(chunks) == 3


def test_chunker_sentence():
    chunker = TextChunker(strategy="sentence", min_chunk_length=1)
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunker.chunk(text, document_id="doc_1")
    assert len(chunks) >= 2


def test_pipeline_process():
    pipe = ProcessingPipeline()
    doc = Document(
        source="test.txt",
        source_type="text",
        content="This is a test document. It has multiple sentences. We need to process it.",
    )
    chunks = pipe.process(doc)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == doc.id
        assert len(chunk.text) > 0


def test_pipeline_basic():
    pipe = ProcessingPipeline()
    doc = Document(
        source="test.txt",
        source_type="text",
        content="Some content here to avoid empty validation. " * 5,
    )
    chunks = pipe.process(doc)
    assert len(chunks) > 0
