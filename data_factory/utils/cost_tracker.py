"""Track token usage and estimated API costs across pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class CostEntry:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CostTracker:
    """Tracks per-model token usage and accumulates estimated costs.

    Pricing is approximate and should be updated for your provider.
    """

    PRICING: Dict[str, Dict[str, float]] = {
        "openai": {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-3.5-turbo": (0.50, 1.50),
        },
        "anthropic": {
            "claude-3-opus": (15.00, 75.00),
            "claude-3-sonnet": (3.00, 15.00),
            "claude-3-haiku": (0.25, 1.25),
        },
        "openrouter": {
            "openai/gpt-4o": (2.50, 10.00),
            "openai/gpt-4o-mini": (0.15, 0.60),
            "openai/gpt-4-turbo": (10.00, 30.00),
            "openai/gpt-3.5-turbo": (0.50, 1.50),
            "anthropic/claude-3-opus": (15.00, 75.00),
            "anthropic/claude-3-sonnet": (3.00, 15.00),
            "anthropic/claude-3-haiku": (0.25, 1.25),
            "meta-llama/llama-3-70b": (0.59, 0.79),
            "mistral/mistral-large": (2.00, 6.00),
        },
    }

    def __init__(self) -> None:
        self.entries: list[CostEntry] = []
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def track(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CostEntry:
        if not self._enabled:
            entry = CostEntry(provider=provider, model=model)
            self.entries.append(entry)
            return entry

        pricing = self.PRICING.get(provider, {}).get(model)
        cost = 0.0
        if pricing:
            prompt_cost = (prompt_tokens / 1000) * pricing[0]
            completion_cost = (completion_tokens / 1000) * pricing[1]
            cost = prompt_cost + completion_cost

        entry = CostEntry(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=round(cost, 6),
        )
        self.entries.append(entry)
        return entry

    @property
    def total_cost(self) -> float:
        return round(sum(e.estimated_cost for e in self.entries), 4)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.entries)

    def summary(self) -> dict:
        by_model: dict = {}
        for e in self.entries:
            key = f"{e.provider}/{e.model}"
            if key not in by_model:
                by_model[key] = {
                    "provider": e.provider,
                    "model": e.model,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost": 0.0,
                    "calls": 0,
                }
            by_model[key]["prompt_tokens"] += e.prompt_tokens
            by_model[key]["completion_tokens"] += e.completion_tokens
            by_model[key]["total_tokens"] += e.total_tokens
            by_model[key]["estimated_cost"] += e.estimated_cost
            by_model[key]["calls"] += 1

        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": len(self.entries),
            "by_model": by_model,
        }

    def reset(self) -> None:
        self.entries.clear()
