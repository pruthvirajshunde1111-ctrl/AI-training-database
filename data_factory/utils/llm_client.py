"""Centralised LLM client — single entry point for all LLM calls.

Supports OpenAI, OpenRouter, and OpenAI-compatible APIs.
Auto-detects provider from key prefix and config. Tracks token usage
and costs via the integrated CostTracker.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from data_factory.utils.cost_tracker import CostTracker
from data_factory.utils.logger import LoggerMixin


class LLMClient(LoggerMixin):
    """Unified LLM client that routes to the correct provider.

    Usage::
        client = LLMClient.from_settings(settings)
        content = client.chat(
            messages=[{"role": "user", "content": "Hello"}],
        )
        content, usage = client.chat_with_usage(...)  # returns token counts
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        default_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        extra_headers: Optional[Dict[str, str]] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.provider = provider
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.cost_tracker = cost_tracker or CostTracker()

    @classmethod
    def from_settings(cls, settings: Any) -> Optional["LLMClient"]:
        api_key = getattr(settings, "llm_api_key", None) or ""
        if not api_key:
            return None

        headers = {}
        base_url = getattr(settings, "llm_base_url", None)

        if api_key.startswith("sk-or-v1-"):
            headers["HTTP-Referer"] = "https://github.com/data-factory"
            headers["X-Title"] = "Data Factory"
            if not base_url:
                base_url = "https://openrouter.ai/api/v1"

        cost_tracker = (
            getattr(settings, "_cost_tracker", None)
            if hasattr(settings, "_cost_tracker")
            else None
        )

        return cls(
            api_key=api_key,
            base_url=base_url,
            provider=(
                getattr(settings, "llm_provider", "openai") or "openai"
            ),
            default_model=getattr(settings, "llm_model", "gpt-4o-mini"),
            temperature=getattr(settings, "llm_temperature", 0.7),
            max_tokens=getattr(settings, "llm_max_tokens", 2048),
            extra_headers=headers,
            cost_tracker=cost_tracker,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> Optional[str]:
        """Send a chat completion request.

        Returns:
            Response text content, or None on failure.
        """
        content, _ = self.chat_with_usage(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return content

    def chat_with_usage(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Send a chat completion request and return (content, usage).

        Usage dict contains ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.

        Returns:
            Tuple of (response text, usage dict or None).
        """
        model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
            "max_tokens": max_tokens or self.max_tokens,
        }

        if response_format == "json" and self.provider in ("openai", "openrouter"):
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        try:
            self.log.debug(
                "LLM call: model=%s, messages=%d, temp=%.1f",
                model,
                len(messages),
                payload["temperature"],
            )
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(self.max_tokens // 10 + 30),
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            usage = None
            if "usage" in data:
                usage = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0),
                }
                if self.cost_tracker:
                    self.cost_tracker.track(
                        provider=self.provider,
                        model=model,
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                    )

            return content, usage

        except requests.HTTPError as e:
            status = e.response.status_code if e.response else 0
            detail = e.response.text[:200] if e.response else str(e)
            self.log.error("LLM API error (HTTP %d): %s", status, detail)
            return None, None
        except requests.Timeout:
            self.log.error(
                "LLM API timeout after %ds", self.max_tokens // 10 + 30
            )
            return None, None
        except Exception as e:
            self.log.error("LLM call failed: %s", e)
            return None, None

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[Any]:
        """Send a chat request and parse the response as JSON."""
        content = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format="json",
        )
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            self.log.warning(
                "LLM response was not valid JSON, attempting fallback parse"
            )
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            for pattern in [r"(\{.*\})", r"(\[.*\])"]:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
            return None

    def is_available(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            return resp.status_code < 500
        except Exception:
            return False
