"""Task Manager — assigns AI tasks to specialist generator modules.

Uses smart templates (recipe cards) to route chunks to the appropriate
generator and collects produced examples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from data_factory.models import TaskExample, TaskType, TextChunk
from data_factory.tasks.templates import TaskTemplateLibrary
from data_factory.utils.logger import LoggerMixin


class TaskManager(LoggerMixin):
    """Coordinates task generation across specialist generators.

    Usage::
        manager = TaskManager(config)
        examples = manager.run(chunks, tasks=["qa", "summarization"])
    """

    GENERATORS: Dict[str, str] = {
        "qa": "data_factory.tasks.qa.QAGenerator",
        "summarization": "data_factory.tasks.summarization.SummarizationGenerator",
        "classification": "data_factory.tasks.classification.ClassificationGenerator",
    }

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self.templates = TaskTemplateLibrary()
        self._generators: Dict[str, Any] = {}

    def _load_generator(self, task_name: str) -> Any:
        if task_name in self._generators:
            return self._generators[task_name]

        import_path = self.GENERATORS.get(task_name)
        if not import_path:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.GENERATORS.keys())}")

        module_path, class_name = import_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls(config=self.config, template=self.templates.get(task_name))
        self._generators[task_name] = instance
        return instance

    def run(
        self,
        chunks: List[TextChunk],
        tasks: Optional[List[str]] = None,
        max_examples_per_task: int = 1000,
    ) -> List[TaskExample]:
        """Run task generation on a list of chunks.

        Args:
            chunks: Text chunks to generate training examples from.
            tasks: Task types to run (defaults to config or all available).
            max_examples_per_task: Maximum number of examples per task.

        Returns:
            Flat list of generated :class:`TaskExample` objects.
        """
        if tasks is None:
            if self.config and hasattr(self.config, "default_tasks"):
                tasks = self.config.default_tasks
            else:
                tasks = list(self.GENERATORS.keys())

        self.log.info("Running tasks: %s (chunks=%d)", tasks, len(chunks))

        all_examples: List[TaskExample] = []
        for task_name in tasks:
            try:
                generator = self._load_generator(task_name)
                examples = generator.generate(
                    chunks, max_examples=max_examples_per_task
                )
                self.log.info(
                    "Task '%s' produced %d examples", task_name, len(examples)
                )
                all_examples.extend(examples)
            except Exception as e:
                self.log.error("Task '%s' failed: %s", task_name, e)

        return all_examples
