"""Track pipeline metadata, run history, and data provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class RunRecord:
    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:8]}")
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    sources: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    documents_loaded: int = 0
    chunks_created: int = 0
    examples_generated: int = 0
    examples_passed_qc: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    status: str = "pending"
    errors: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class MetadataTracker:
    """Tracks pipeline run history and data provenance."""

    def __init__(self) -> None:
        self._runs: List[RunRecord] = []
        self._current: Optional[RunRecord] = None

    def start_run(
        self, sources: Optional[List[str]] = None, tasks: Optional[List[str]] = None
    ) -> RunRecord:
        record = RunRecord(
            sources=sources or [],
            tasks=tasks or [],
        )
        self._current = record
        self._runs.append(record)
        return record

    def finish_run(
        self,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> RunRecord:
        if self._current is None:
            raise RuntimeError("No active run to finish")
        self._current.finished_at = datetime.utcnow()
        self._current.status = status
        if error:
            self._current.errors.append(error)
        return self._current

    @property
    def current(self) -> Optional[RunRecord]:
        return self._current

    @property
    def runs(self) -> List[RunRecord]:
        return list(self._runs)

    def summary(self) -> dict:
        return {
            "total_runs": len(self._runs),
            "recent_runs": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "sources": r.sources,
                    "tasks": r.tasks,
                    "examples": r.examples_generated,
                    "passed_qc": r.examples_passed_qc,
                    "cost": r.total_cost,
                }
                for r in self._runs[-5:]
            ],
        }
