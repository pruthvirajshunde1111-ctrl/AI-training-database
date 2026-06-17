"""Classification generator — produces text classification training examples."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from data_factory.models import TaskExample, TaskTemplate, TaskType, TextChunk
from data_factory.utils.llm_client import LLMClient
from data_factory.utils.logger import LoggerMixin

DEFAULT_CATEGORIES = [
    "technology",
    "science",
    "business",
    "health",
    "education",
    "entertainment",
    "politics",
    "sports",
    "environment",
    "lifestyle",
]


class ClassificationGenerator(LoggerMixin):
    """Generates classification training examples from text chunks.

    Produces multi-label classification examples with configurable
    category taxonomies.
    """

    def __init__(
        self, config: Optional[Any] = None, template: Optional[TaskTemplate] = None
    ) -> None:
        self.config = config
        self.template = template
        self._llm = LLMClient.from_settings(config) if config else None
        self.categories = DEFAULT_CATEGORIES

    def generate(
        self, chunks: List[TextChunk], max_examples: int = 1000
    ) -> List[TaskExample]:
        examples: List[TaskExample] = []

        for chunk in chunks:
            if len(examples) >= max_examples:
                break

            try:
                chunk_examples = self._generate_classifications(chunk)
                examples.extend(chunk_examples)
            except Exception as e:
                self.log.warning(
                    "Classification failed for chunk %s: %s", chunk.id, e
                )

        self.log.info("Generated %d classification examples", len(examples))
        return examples

    def _generate_classifications(self, chunk: TextChunk) -> List[TaskExample]:
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
        categories_str = ", ".join(self.categories)
        system = (
            f"You are a classification data generator. Classify the given text into "
            f"one or more of these categories: {categories_str}. "
            "Return a JSON object with 'labels' (list of category strings) and "
            "'justification' (brief explanation)."
        )

        data = self._llm.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Text:\n{chunk.text}"},
            ],
            temperature=0.3,
        )

        if not data:
            self.log.warning("LLM returned no data, falling back to rule-based")
            return self._generate_rule_based(chunk)

        labels = data.get("labels", data.get("categories", data.get("classes", [])))
        justification = data.get("justification", "")

        if labels:
            return [
                TaskExample(
                    task_type=TaskType.CLASSIFICATION,
                    task_name="classification",
                    input=f"Classify the following text into categories: {categories_str}\n\n{chunk.text}",
                    expected_output=json.dumps(
                        {"labels": labels, "justification": justification},
                        ensure_ascii=False,
                    ),
                    metadata={"num_labels": len(labels), "method": "llm"},
                )
            ]

        return self._generate_rule_based(chunk)

    def _generate_rule_based(self, chunk: TextChunk) -> List[TaskExample]:
        examples: List[TaskExample] = []
        text_lower = chunk.text.lower()

        keyword_map: Dict[str, List[str]] = {
            "technology": ["computer", "software", "ai", "data", "digital", "tech", "algorithm", "code"],
            "science": ["study", "research", "experiment", "scientific", "theory", "analysis"],
            "business": ["company", "market", "revenue", "startup", "investment", "economy"],
            "health": ["health", "medical", "patient", "treatment", "disease", "doctor", "clinical"],
            "education": ["student", "learning", "school", "teaching", "education", "curriculum"],
            "entertainment": ["movie", "music", "film", "game", "show", "artist", "entertainment"],
            "politics": ["government", "election", "policy", "political", "president", "law"],
            "sports": ["sport", "game", "team", "player", "match", "tournament", "athlete"],
            "environment": ["climate", "environment", "sustainable", "energy", "pollution", "green"],
            "lifestyle": ["food", "fashion", "travel", "fitness", "recipe", "lifestyle"],
        }

        detected = []
        for category, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(category)

        if detected:
            examples.append(
                TaskExample(
                    task_type=TaskType.CLASSIFICATION,
                    task_name="classification",
                    input=f"Classify the following text into categories: {', '.join(self.categories)}\n\n{chunk.text}",
                    expected_output=json.dumps(
                        {"labels": detected, "justification": "Keyword-based classification"},
                        ensure_ascii=False,
                    ),
                    metadata={"num_labels": len(detected), "method": "rule_based"},
                )
            )

        if not detected:
            random_label = random.choice(self.categories)
            examples.append(
                TaskExample(
                    task_type=TaskType.CLASSIFICATION,
                    task_name="classification",
                    input=f"Classify the following text into categories: {', '.join(self.categories)}\n\n{chunk.text}",
                    expected_output=json.dumps({"labels": [random_label], "justification": "Random assignment"}, ensure_ascii=False),
                    metadata={"num_labels": 1, "method": "rule_based_random"},
                )
            )

        return examples
