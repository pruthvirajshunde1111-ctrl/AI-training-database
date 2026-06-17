"""Summarization generator — produces extractive and abstractive summaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from data_factory.models import TaskExample, TaskTemplate, TaskType, TextChunk
from data_factory.utils.llm_client import LLMClient
from data_factory.utils.logger import LoggerMixin


class SummarizationGenerator(LoggerMixin):
    """Generates summarization training examples from text chunks.

    Produces:
      - Extractive summaries (key sentences)
      - Abstractive summaries (paraphrased)
      - Bullet-point summaries
    """

    def __init__(
        self, config: Optional[Any] = None, template: Optional[TaskTemplate] = None
    ) -> None:
        self.config = config
        self.template = template
        self._llm = LLMClient.from_settings(config) if config else None

    def generate(
        self, chunks: List[TextChunk], max_examples: int = 1000
    ) -> List[TaskExample]:
        examples: List[TaskExample] = []

        for chunk in chunks:
            if len(examples) >= max_examples:
                break

            try:
                chunk_examples = self._generate_summaries(chunk)
                examples.extend(chunk_examples)
            except Exception as e:
                self.log.warning("Summarization failed for chunk %s: %s", chunk.id, e)

        self.log.info("Generated %d summarization examples", len(examples))
        return examples

    def _generate_summaries(self, chunk: TextChunk) -> List[TaskExample]:
        examples: List[TaskExample] = []

        if self._llm:
            llm_examples = self._generate_llm(chunk)
            examples.extend(llm_examples)
        else:
            rule_examples = self._generate_rule_based(chunk)
            examples.extend(rule_examples)

        for ex in examples:
            ex.source_chunk_id = chunk.id
            ex.source_document_id = chunk.document_id

        return examples

    def _generate_llm(self, chunk: TextChunk) -> List[TaskExample]:
        system = (
            "You are a summarization data generator. Create three summaries of the given text: "
            "1) extractive (key sentences verbatim), "
            "2) abstractive (paraphrased in your own words), "
            "3) bullet points (key takeaways). "
            "Return JSON with keys: 'extractive', 'abstractive', 'bullet_points'."
        )

        data = self._llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Text:\n{chunk.text}"},
            ],
            temperature=0.5,
        )

        if not data:
            self.log.warning("LLM returned no data, falling back to rule-based")
            return self._generate_rule_based(chunk)

        examples = []
        summary_types = {
            "extractive": "Extractive summary of the passage",
            "abstractive": "Abstractive summary of the passage",
            "bullet_points": "Bullet-point summary of the passage",
        }

        for key, instruction in summary_types.items():
            if key in data and data[key]:
                examples.append(
                    TaskExample(
                        task_type=TaskType.SUMMARIZATION,
                        task_name="summarization",
                        input=f"{instruction}\n\n{chunk.text}",
                        expected_output=str(data[key]),
                        metadata={"summary_type": key, "method": "llm"},
                    )
                )
        return examples if examples else self._generate_rule_based(chunk)

    def _generate_rule_based(self, chunk: TextChunk) -> List[TaskExample]:
        examples: List[TaskExample] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return examples

        extractive = " ".join(sentences[: min(3, len(sentences))])
        examples.append(
            TaskExample(
                task_type=TaskType.SUMMARIZATION,
                task_name="summarization",
                input=f"Extractive summary of the passage\n\n{chunk.text}",
                expected_output=extractive,
                metadata={"summary_type": "extractive", "method": "rule_based"},
            )
        )

        bullets = "\n".join(f"- {s}" for s in sentences[: min(5, len(sentences))])
        examples.append(
            TaskExample(
                task_type=TaskType.SUMMARIZATION,
                task_name="summarization",
                input=f"Bullet-point summary of the passage\n\n{chunk.text}",
                expected_output=bullets,
                metadata={"summary_type": "bullet_points", "method": "rule_based"},
            )
        )

        return examples
