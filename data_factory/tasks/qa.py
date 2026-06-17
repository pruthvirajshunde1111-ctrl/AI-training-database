"""QA pair generator — produces question-answer training examples."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from data_factory.models import TaskExample, TaskTemplate, TaskType, TextChunk
from data_factory.utils.llm_client import LLMClient
from data_factory.utils.logger import LoggerMixin


class QAGenerator(LoggerMixin):
    """Generates question-answer pairs from text chunks.

    Uses either an LLM provider or rule-based extraction to produce
    diverse QA pairs covering factual, inferential, and analytical levels.
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
        """Generate QA pairs from the provided chunks."""
        examples: List[TaskExample] = []
        max_per_chunk = 3

        if self.config and hasattr(self.config, "max_examples_per_chunk"):
            max_per_chunk = self.config.max_examples_per_chunk

        for chunk in chunks:
            if len(examples) >= max_examples:
                break

            try:
                chunk_examples = self._generate_from_chunk(chunk, max_per_chunk)
                examples.extend(chunk_examples)
            except Exception as e:
                self.log.warning("QA gen failed for chunk %s: %s", chunk.id, e)

        self.log.info("Generated %d QA examples", len(examples))
        return examples

    def _generate_from_chunk(
        self, chunk: TextChunk, max_pairs: int = 3
    ) -> List[TaskExample]:
        examples: List[TaskExample] = []

        if self._llm:
            examples = self._generate_llm(chunk, max_pairs)
        else:
            examples = self._generate_rule_based(chunk, max_pairs)

        for ex in examples:
            ex.source_chunk_id = chunk.id
            ex.source_document_id = chunk.document_id

        return examples

    def _generate_llm(self, chunk: TextChunk, max_pairs: int) -> List[TaskExample]:
        system = (
            "You are a data generation assistant. Generate diverse question-answer "
            "pairs from the given text. Cover factual, inferential, and analytical "
            f"questions. Return a JSON list of objects with 'question' and 'answer' keys. Generate up to {max_pairs} pairs."
        )

        data = self._llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Text:\n{chunk.text}"},
            ],
            temperature=0.7,
        )

        if not data:
            self.log.warning("LLM returned no data, falling back to rule-based")
            return self._generate_rule_based(chunk, max_pairs)

        pairs = data if isinstance(data, list) else data.get("pairs", data.get("questions", []))
        examples = []
        for pair in pairs[:max_pairs]:
            q = pair.get("question", pair.get("q", ""))
            a = pair.get("answer", pair.get("a", ""))
            if q and a:
                examples.append(
                    TaskExample(
                        task_type=TaskType.QA,
                        task_name="qa_generation",
                        input=q,
                        expected_output=a,
                        metadata={"method": "llm", "model": self._llm.default_model if self._llm else "unknown"},
                    )
                )
        return examples if examples else self._generate_rule_based(chunk, max_pairs)

    def _generate_rule_based(self, chunk: TextChunk, max_pairs: int) -> List[TaskExample]:
        examples: List[TaskExample] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)

        for i, sent in enumerate(sentences[: max_pairs * 3]):
            if len(examples) >= max_pairs:
                break
            sent = sent.strip()
            if len(sent) < 20:
                continue

            if "?" in sent:
                q, _, rest = sent.partition("?")
                a = rest.strip() or "See text above."
                examples.append(
                    TaskExample(
                        task_type=TaskType.QA,
                        task_name="qa_generation",
                        input=q.strip() + "?",
                        expected_output=a,
                        metadata={"method": "rule_based", "type": "extracted_question"},
                    )
                )
            else:
                words = sent.split()
                if len(words) > 5:
                    question = f"What is mentioned about '{words[min(3, len(words)-1)]}' in the text?"
                    examples.append(
                        TaskExample(
                            task_type=TaskType.QA,
                            task_name="qa_generation",
                            input=question,
                            expected_output=sent,
                            metadata={"method": "rule_based", "type": "keyword_question"},
                        )
                    )

        return examples
