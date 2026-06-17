"""DataFactory — the General Manager that orchestrates the entire pipeline.

Coordinates document loading, processing, task generation, quality
evaluation, and dataset export in a single cohesive workflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from data_factory.config import FactorySettings
from data_factory.loaders import FileLoader, PDFLoader, URLLoader
from data_factory.loaders.base import BaseLoader
from data_factory.models import Dataset, Document, TaskExample, TextChunk
from data_factory.processors.pipeline import ProcessingPipeline
from data_factory.quality.pipeline import QualityPipeline
from data_factory.tasks.manager import TaskManager
from data_factory.utils.cost_tracker import CostTracker
from data_factory.utils.logger import LoggerMixin, get_logger
from data_factory.utils.metadata import MetadataTracker


class DataFactory(LoggerMixin):
    """The main orchestrator for the AI training data generation pipeline.

    Typical usage::

        from data_factory import DataFactory
        from data_factory.config import FactorySettings

        settings = FactorySettings(
            llm_api_key="sk-...",
            chunk_size=512,
            output_dir="./my_dataset",
        )
        factory = DataFactory(settings)
        dataset = factory.run(
            sources=["doc.pdf", "https://example.com"],
            tasks=["qa", "summarization"],
        )
        dataset.export("./training_data.jsonl")
        print(factory.summary())
    """

    def __init__(
        self,
        settings: Optional[Union[FactorySettings, Dict[str, Any]]] = None,
    ) -> None:
        if settings is None:
            self.settings = FactorySettings()
        elif isinstance(settings, dict):
            self.settings = FactorySettings(**settings)
        else:
            self.settings = settings

        # Auto-load LLM API key from .env / environment if not set
        import os as _os

        if not self.settings.llm_api_key:
            for env_var in ["OPENAI_API_KEY", "LLM_API_KEY", "OPENROUTER_API_KEY"]:
                val = _os.environ.get(env_var)
                if val:
                    self.settings.llm_api_key = val
                    break

        self._logger = get_logger(
            self.__class__.__name__,
            level=self.settings.log_level,
            log_file=self.settings.log_file,
        )

        # Sub-systems
        from data_factory.utils.llm_client import LLMClient

        self._loaders: Dict[str, BaseLoader] = {}
        self.pipeline = ProcessingPipeline(self.settings)
        self.task_manager = TaskManager(self.settings)
        self.quality = QualityPipeline(self.settings)
        self.llm_client = LLMClient.from_settings(self.settings)

        # Tracking
        self.cost_tracker = CostTracker()
        self.cost_tracker.enabled = self.settings.track_costs
        self.metadata_tracker = MetadataTracker()

        # Re-create LLM client with cost tracker wired in
        if self.llm_client:
            self.llm_client.cost_tracker = self.cost_tracker

        # State
        self._documents: List[Document] = []
        self._chunks: List[TextChunk] = []
        self._dataset: Optional[Dataset] = None

        llm_status = (
            f"provider={self.settings.llm_provider}, model={self.settings.llm_model}"
            if self.llm_client
            else "DISABLED (rule-based fallback)"
        )
        self.log.info(
            "DataFactory initialized (v%s, chunk_size=%d, tasks=%s, llm=%s)",
            self.settings.version,
            self.settings.chunk_size,
            self.settings.default_tasks,
            llm_status,
        )

    # ── Document Loading ─────────────────────────────────────────────

    def _get_loader(self, source: str) -> BaseLoader:
        """Select the appropriate loader based on the source type."""
        source_lower = source.lower().strip()

        if source_lower.startswith(("http://", "https://")):
            loader_key = "url"
        elif source_lower.endswith(".pdf"):
            loader_key = "pdf"
        else:
            loader_key = "file"

        if loader_key not in self._loaders:
            if loader_key == "url":
                self._loaders[loader_key] = URLLoader(self.settings)
            elif loader_key == "pdf":
                self._loaders[loader_key] = PDFLoader(self.settings)
            else:
                self._loaders[loader_key] = FileLoader(self.settings)

        return self._loaders[loader_key]

    def load_sources(
        self, sources: List[str]
    ) -> List[Document]:
        """Load documents from a list of sources (files, URLs, PDFs).

        Each source is routed to the correct loader automatically.
        """
        self.log.info("Loading %d source(s)...", len(sources))
        all_docs: List[Document] = []

        for source in sources:
            try:
                loader = self._get_loader(source)
                docs = loader.load(source)
                all_docs.extend(docs)
                self.log.info(
                    "Loaded %d document(s) from: %s", len(docs), source
                )
            except Exception as e:
                self.log.error("Failed to load '%s': %s", source, e)

        self._documents = all_docs
        self.log.info(
            "Total documents loaded: %d", len(all_docs)
        )
        return all_docs

    # ── Processing ───────────────────────────────────────────────────

    def process_documents(
        self, documents: Optional[List[Document]] = None
    ) -> List[TextChunk]:
        """Clean and chunk documents into processable pieces."""
        docs = documents if documents is not None else self._documents

        if not docs:
            self.log.warning("No documents to process")
            return []

        self.log.info("Processing %d document(s)...", len(docs))
        chunks = self.pipeline.process_batch(docs)
        self._chunks = chunks
        self.log.info("Created %d text chunk(s)", len(chunks))
        return chunks

    # ── Task Generation ──────────────────────────────────────────────

    def generate_examples(
        self,
        chunks: Optional[List[TextChunk]] = None,
        tasks: Optional[List[str]] = None,
    ) -> List[TaskExample]:
        """Generate training examples from text chunks using task generators."""
        chunks = chunks if chunks is not None else self._chunks
        tasks = tasks or self.settings.default_tasks

        if not chunks:
            self.log.warning("No chunks available for generation")
            return []

        self.log.info(
            "Generating examples (tasks=%s, chunks=%d)...",
            tasks,
            len(chunks),
        )
        examples = self.task_manager.run(
            chunks=chunks,
            tasks=tasks,
            max_examples_per_task=self.settings.max_examples_per_task,
        )
        self.log.info("Generated %d example(s)", len(examples))
        return examples

    # ── Quality Control ──────────────────────────────────────────────

    def evaluate_quality(
        self, examples: Optional[List[TaskExample]] = None
    ) -> Tuple[List[TaskExample], List[TaskExample], List[Any]]:
        """Run quality checks on examples.

        Returns:
            Tuple of (passed_examples, failed_examples, quality_reports).
        """
        examples = examples if examples is not None else (
            self._dataset.examples if self._dataset else []
        )

        if not examples:
            self.log.warning("No examples to evaluate")
            return [], [], []

        if not self.settings.quality_enabled:
            self.log.info("Quality checks disabled, skipping")
            return examples, [], []

        self.log.info("Evaluating quality of %d example(s)...", len(examples))
        from data_factory.models import QualityReport
        reports: List[QualityReport] = self.quality.evaluate_batch(examples)

        passed: List[TaskExample] = []
        failed: List[TaskExample] = []
        example_map = {ex.id: ex for ex in examples}

        for report in reports:
            if report.passed:
                ex = example_map.get(report.example_id)
                if ex:
                    ex.quality_scores = report.scores
                    passed.append(ex)
            else:
                ex = example_map.get(report.example_id)
                if ex:
                    ex.quality_scores = report.scores
                    failed.append(ex)

        self.log.info(
            "Quality check: %d passed, %d failed",
            len(passed),
            len(failed),
        )
        return passed, failed, reports

    # ── Dataset Assembly ─────────────────────────────────────────────

    def assemble_dataset(
        self,
        examples: Optional[List[TaskExample]] = None,
        name: Optional[str] = None,
        quality_reports: Optional[List[Any]] = None,
    ) -> Dataset:
        """Assemble training examples into a final Dataset object."""
        examples = examples if examples is not None else []

        dataset = Dataset(
            name=name or f"dataset_{int(time.time())}",
            examples=examples,
            metadata={
                "version": self.settings.version,
                "chunk_size": self.settings.chunk_size,
                "total_documents": len(self._documents),
                "total_chunks": len(self._chunks),
                "total_examples": len(examples),
                "tasks": self.settings.default_tasks,
            },
            quality_reports=quality_reports or [],
        )

        self._dataset = dataset
        self.log.info(
            "Dataset assembled: %d examples", dataset.size
        )
        return dataset

    # ── Full Pipeline Run ────────────────────────────────────────────

    def run(
        self,
        sources: List[str],
        tasks: Optional[List[str]] = None,
        run_name: Optional[str] = None,
        export_path: Optional[str] = None,
        skip_quality: bool = False,
        max_chunks: Optional[int] = None,
    ) -> Dataset:
        """Execute the full data factory pipeline end-to-end.

        Args:
            sources: List of file paths, URLs, or PDF paths to load.
            tasks: Task types to generate (qa, summarization, classification).
            run_name: Optional name for this run/dataset.
            export_path: Optional path to export the final dataset.
            skip_quality: If True, skip quality evaluation.

        Returns:
            The assembled :class:`Dataset`.
        """
        tasks = tasks or self.settings.default_tasks
        run = self.metadata_tracker.start_run(
            sources=sources, tasks=tasks
        )
        start_time = time.time()

        self.log.info("=" * 60)
        self.log.info("Data Factory Run Started")
        self.log.info("  Sources: %s", sources)
        self.log.info("  Tasks:   %s", tasks)
        self.log.info("=" * 60)

        # 1. Load
        docs = self.load_sources(sources)
        run.documents_loaded = len(docs)

        # 2. Process
        chunks = self.process_documents(docs)
        run.chunks_created = len(chunks)

        # 2b. Optionally limit chunks for dashboard responsiveness
        limit = max_chunks or self.settings.max_chunks or 0
        if limit > 0 and len(chunks) > limit:
            self.log.info(
                "Limiting to %d chunks (of %d) for task generation",
                limit, len(chunks),
            )
            chunks = chunks[:limit]

        # 3. Generate
        examples = self.generate_examples(chunks, tasks)
        run.examples_generated = len(examples)

        # 4. Quality
        quality_reports: List[Any] = []
        if not skip_quality and self.settings.quality_enabled and examples:
            passed, failed, quality_reports = self.evaluate_quality(examples)
            run.examples_passed_qc = len(passed)
            final_examples = passed
            self.log.info(
                "QC Results: %d passed / %d failed",
                len(passed),
                len(failed),
            )
        else:
            final_examples = examples
            run.examples_passed_qc = len(examples)

        # 5. Assemble
        dataset = self.assemble_dataset(
            examples=final_examples,
            name=run_name or f"run_{run.run_id}",
            quality_reports=quality_reports,
        )

        # 6. Export if requested
        if export_path:
            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fmt = "json" if export_path.endswith(".json") else "jsonl"
            dataset.export(str(path), format=fmt)
            self.log.info("Dataset exported to: %s", path)

        # 7. Finalize tracking
        elapsed = time.time() - start_time
        run.total_tokens_used = self.cost_tracker.total_tokens
        run.total_cost = self.cost_tracker.total_cost
        self.metadata_tracker.finish_run(status="completed")

        self.log.info("=" * 60)
        self.log.info("Run Complete: %s", run.run_id)
        self.log.info("  Documents:  %d", run.documents_loaded)
        self.log.info("  Chunks:     %d", run.chunks_created)
        self.log.info("  Examples:   %d", run.examples_generated)
        self.log.info("  Passed QC:  %d", run.examples_passed_qc)
        self.log.info("  Cost:       $%.4f", run.total_cost)
        self.log.info("  Time:       %.2fs", elapsed)
        self.log.info("=" * 60)

        return dataset

    # ── Observability ────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the factory's state and history."""
        return {
            "version": self.settings.version,
            "status": "ready",
            "documents_loaded": len(self._documents),
            "chunks_created": len(self._chunks),
            "dataset_size": self._dataset.size if self._dataset else 0,
            "total_cost": self.cost_tracker.total_cost,
            "total_tokens": self.cost_tracker.total_tokens,
            "runs": self.metadata_tracker.summary(),
            "cost_breakdown": self.cost_tracker.summary(),
        }

    def export_run_log(self, path: str) -> str:
        """Export run metadata and cost logs to a JSON file."""
        data = {
            "summary": self.summary(),
            "settings": self.settings.model_dump(),
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        self.log.info("Run log exported to: %s", path)
        return str(path_obj)

    @property
    def dataset(self) -> Optional[Dataset]:
        return self._dataset

    @property
    def documents(self) -> List[Document]:
        return self._documents

    @property
    def chunks(self) -> List[TextChunk]:
        return self._chunks
