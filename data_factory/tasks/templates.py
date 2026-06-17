"""Template library — the recipe cards for each task type.

Each template defines the system prompt, user prompt template, and
constraints that specialist generators use to produce training examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_factory.models import TaskTemplate, TaskType


class TaskTemplateLibrary:
    """Central registry of task templates / recipe cards."""

    def __init__(self) -> None:
        self._templates: Dict[str, TaskTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            TaskTemplate(
                name="qa_generation",
                task_type=TaskType.QA,
                description="Generate question-answer pairs from text passages",
                system_prompt=(
                    "You are a data generation assistant. Your task is to create "
                    "high-quality question-answer pairs from the given text. "
                    "Questions should be diverse, answerable from the text, and "
                    "range from factual to inferential."
                ),
                user_prompt_template=(
                    "Based on the following passage, generate {num_examples} "
                    "question-answer pairs. Each pair should be a JSON object "
                    'with "question" and "answer" fields.\n\nPassage:\n{text}'
                ),
                instructions="Generate diverse QA pairs that cover different aspects of the text",
                constraints={
                    "num_examples": 3,
                    "min_question_length": 10,
                    "max_question_length": 200,
                    "min_answer_length": 5,
                    "max_answer_length": 500,
                },
            )
        )
        self.register(
            TaskTemplate(
                name="summarization",
                task_type=TaskType.SUMMARIZATION,
                description="Generate text summaries at varying levels of detail",
                system_prompt=(
                    "You are a summarization data generator. Create summaries "
                    "of the given text at different lengths and abstraction levels. "
                    "Include extractive (key sentence) and abstractive variants."
                ),
                user_prompt_template=(
                    "Summarize the following passage in {summary_type} format. "
                    "Target length: {target_length} words.\n\nPassage:\n{text}"
                ),
                instructions="Generate extractive and abstractive summaries",
                constraints={
                    "summary_types": ["extractive", "abstractive", "bullet_points"],
                    "target_lengths": [50, 100, 200],
                },
            )
        )
        self.register(
            TaskTemplate(
                name="classification",
                task_type=TaskType.CLASSIFICATION,
                description="Generate text classification examples with labels",
                system_prompt=(
                    "You are a classification data generator. Given the text below, "
                    "generate classification examples with appropriate labels. "
                    "Cover multiple categories and confidence levels."
                ),
                user_prompt_template=(
                    "Classify the following text into one or more of these categories: "
                    "{categories}. Provide the label(s) and a brief justification.\n\n"
                    "Text:\n{text}"
                ),
                instructions="Generate multi-label classification examples",
                constraints={
                    "num_classes": 5,
                    "require_justification": True,
                },
            )
        )
        self.register(
            TaskTemplate(
                name="extraction",
                task_type=TaskType.EXTRACTION,
                description="Extract structured information from unstructured text",
                system_prompt=(
                    "You are an information extraction data generator. Extract "
                    "structured fields from the given text according to the schema."
                ),
                user_prompt_template=(
                    "Extract the following fields from the text: {fields}. "
                    "Return the result as a JSON object.\n\nText:\n{text}"
                ),
                instructions="Generate structured extraction examples",
                constraints={
                    "fields": ["entities", "dates", "relationships"],
                    "output_format": "json",
                },
            )
        )

    def register(self, template: TaskTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[TaskTemplate]:
        return self._templates.get(name)

    def get_by_type(self, task_type: TaskType) -> List[TaskTemplate]:
        return [t for t in self._templates.values() if t.task_type == task_type]

    def list_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "task_type": t.task_type.value,
                "description": t.description,
                "version": t.version,
            }
            for t in self._templates.values()
        ]

    @property
    def count(self) -> int:
        return len(self._templates)
