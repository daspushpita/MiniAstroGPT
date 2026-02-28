from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResult:
    text: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: int | None = None
    tool_calls: list[dict[str, Any]] | None = None
    raw: dict[str, Any] | None = None


class LLMClient(ABC):
    """Provider-agnostic contract for model calls."""
    
    @abstractmethod
    def generate(
        self,
        stage: str,
        prompt: str | None = None,
        *,
        messages: list[dict[str, object]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        raise NotImplementedError
        
