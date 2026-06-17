"""Data Factory - AI Training Dataset Generation Pipeline.

A modular, production-ready pipeline for creating high-quality
training datasets for LLM-based tasks such as chatbots,
summarization, and classification.

The system follows a factory architecture:
  - Document Loaders: Ingest raw data from PDFs, URLs, local files
  - Text Processing: Clean, chunk, and prepare raw text
  - Task Management: Generate QA pairs, summaries, classifications
  - Quality Control: Evaluate for toxicity, bias, diversity, coherence
  - Export: Deliver ready-to-use datasets

Typical usage:
    from data_factory import DataFactory
    from data_factory.config import settings

    factory = DataFactory(settings)
    dataset = factory.run(
        sources=["document.pdf", "https://example.com"],
        tasks=["qa", "summarization"]
    )
    dataset.export("training_data.jsonl")
"""

__version__ = "0.1.0"
__author__ = "Data Factory Team"

from data_factory.bot import DataFactory
from data_factory.models import (
    Document,
    TextChunk,
    TaskTemplate,
    TaskExample,
    QualityReport,
    Dataset,
)
from data_factory.config import FactorySettings

__all__ = [
    "DataFactory",
    "FactorySettings",
    "Document",
    "TextChunk",
    "TaskTemplate",
    "TaskExample",
    "QualityReport",
    "Dataset",
]
