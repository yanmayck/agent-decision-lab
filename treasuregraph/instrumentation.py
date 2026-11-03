"""Resource instrumentation and cost estimation for TreasureGraph experiments.

Provides clean separation between agent decision logic and resource tracking:
1. input_tokens: API-reported prompt tokens.
2. output_tokens: API-reported completion tokens.
3. reasoning_tokens: API-reported thinking/reasoning tokens when available (null when absent).
4. tool_result_tokens: Heuristic estimate of tool result tokens, identified as an estimate
   and never double-counted into API billable tokens.
5. tool_calls: Total count of tool actions executed.
6. api_calls: Total count of model API requests.
7. estimated_cost_usd: Estimated cost considering configurable provider pricing, distinguishing
   reasoning tokens when necessary and avoiding double-counting (null when rates unknown).
8. wall_clock_seconds: Total execution duration.
Individual tool executions record their exact size in bytes (result_bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any


def estimate_tokens(text: str) -> int:
    """Heuristic estimate of tokens for arbitrary text (e.g. tool results).

    Uses the standard heuristic of ~4 bytes per token for UTF-8 encoded text.
    Labeled explicitly as an estimate and never added to API billable tokens.
    """
    if not text:
        return 0
    num_bytes = len(text.encode("utf-8"))
    return max(1, math.ceil(num_bytes / 4))


def parse_api_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Parses token usage from OpenAI or Gemini API response, verifying provider semantics.

    Distinguishes whether reasoning tokens are already included in output_tokens (OpenAI
    convention) or reported separately (e.g. Google Gemini OpenAI compatibility layer).

    Returns:
        dict with:
        - prompt_tokens (int)
        - completion_tokens (int)
        - reasoning_tokens (int | None)
        - reasoning_included_in_output (bool)
        - total_tokens (int)
    """
    if not usage or not isinstance(usage, dict):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": None,
            "reasoning_included_in_output": True,
            "total_tokens": 0,
        }

    p = int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0)
    c = int(usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0)
    total = usage.get("total_tokens") or usage.get("totalTokenCount")
    total_val = int(total) if total is not None else (p + c)

    details = usage.get("completion_tokens_details") or {}
    explicit_r = details.get("reasoning_tokens")
    if explicit_r is None:
        explicit_r = usage.get("thoughtsTokenCount") or usage.get("thinkingTokenCount")

    if explicit_r is not None:
        r: int | None = int(explicit_r)
        # If total_val exceeds (p + c) by at least r, then r was NOT included in c
        if total is not None and total_val >= (p + c + r) and r > 0:
            included = False
        elif total is not None and total_val == (p + c):
            included = True
        else:
            included = True
    elif total is not None and total_val > (p + c):
        # Gemini OpenAI compatibility endpoint: total_tokens = prompt + completion + thinking,
        # but reasoning_tokens is omitted from completion_tokens_details.
        r = total_val - (p + c)
        included = False
    else:
        r = None
        included = True

    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "reasoning_tokens": r,
        "reasoning_included_in_output": included,
        "total_tokens": total_val,
    }


@dataclass(frozen=True)
class PricingConfig:
    """Configurable pricing in USD per 1,000,000 (1M) tokens.

    Attributes:
        input_cost_per_million: Price in USD per 1M input (prompt) tokens.
            If None, estimated_cost_usd evaluates to null (None).
        output_cost_per_million: Price in USD per 1M output (completion) tokens.
            If None, estimated_cost_usd evaluates to null (None).
        reasoning_cost_per_million: Optional separate price for reasoning tokens.
            If None, billed at output_cost_per_million.
        reasoning_included_in_output: Optional override boolean.
            If True, output_tokens already covers reasoning tokens.
            If False, reasoning tokens are billed additionally.
            If None, detected dynamically from provider API usage.
    """

    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    reasoning_cost_per_million: float | None = None
    reasoning_included_in_output: bool | None = None

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None = None,
        reasoning_included_in_output: bool = True,
    ) -> float | None:
        """Calculates estimated cost in USD, strictly avoiding double counting.

        Returns None if input or output pricing is not configured.
        """
        if self.input_cost_per_million is None or self.output_cost_per_million is None:
            return None

        included = (
            self.reasoning_included_in_output
            if self.reasoning_included_in_output is not None
            else reasoning_included_in_output
        )

        r_tokens = reasoning_tokens or 0

        # Base prompt/input cost
        cost = (input_tokens / 1_000_000.0) * self.input_cost_per_million

        # Output and reasoning cost
        if included:
            # output_tokens already contains reasoning_tokens
            cost += (output_tokens / 1_000_000.0) * self.output_cost_per_million
            if self.reasoning_cost_per_million is not None and r_tokens > 0:
                diff_rate = self.reasoning_cost_per_million - self.output_cost_per_million
                cost += (r_tokens / 1_000_000.0) * diff_rate
        else:
            # output_tokens does NOT include reasoning_tokens
            cost += (output_tokens / 1_000_000.0) * self.output_cost_per_million
            if r_tokens > 0:
                r_rate = (
                    self.reasoning_cost_per_million
                    if self.reasoning_cost_per_million is not None
                    else self.output_cost_per_million
                )
                cost += (r_tokens / 1_000_000.0) * r_rate

        return round(cost, 6)


@dataclass
class ToolExecutionRecord:
    """Record of an individual tool call and its resource footprint."""

    step: int
    action: str
    arguments: dict[str, Any]
    result_bytes: int
    estimated_tool_result_tokens: int
    duration_seconds: float
    cost_units: int = 0
    is_parallel: bool = False


class ResourceTracker:
    """Tracks token consumption, tool footprints, latency, and costs.

    Decoupled from agent decision logic to maintain scientific instrumentation integrity.
    """

    def __init__(self, pricing: PricingConfig | None = None) -> None:
        self.pricing = pricing
        self.start_time: float | None = None
        self.end_time: float | None = None

        self.api_calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.reasoning_tokens: int | None = None
        self.estimated_tool_result_tokens: int = 0
        self._any_reasoning_separate: bool = False

        self.tool_executions: list[ToolExecutionRecord] = []

    def start(self) -> None:
        """Marks the start of the experimental run."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Marks the end of the experimental run."""
        self.end_time = time.perf_counter()

    @property
    def wall_clock_seconds(self) -> float:
        """Total wall-clock duration of the tracked run."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return round(end - self.start_time, 4)

    def record_api_call(
        self,
        usage: dict[str, Any],
        duration_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Records an API request and its usage metadata reported by the provider."""
        self.api_calls += 1
        parsed = parse_api_usage(usage)

        self.input_tokens += parsed["prompt_tokens"]
        self.output_tokens += parsed["completion_tokens"]

        if parsed["reasoning_tokens"] is not None:
            if self.reasoning_tokens is None:
                self.reasoning_tokens = parsed["reasoning_tokens"]
            else:
                self.reasoning_tokens += parsed["reasoning_tokens"]

        if not parsed["reasoning_included_in_output"]:
            self._any_reasoning_separate = True

        return parsed

    def record_tool_execution(
        self,
        step: int,
        action: str,
        arguments: dict[str, Any],
        result_text: str,
        duration_seconds: float,
        cost_units: int = 0,
        is_parallel: bool = False,
    ) -> ToolExecutionRecord:
        """Records tool execution, calculating exact byte length and estimated tokens.

        Estimates tokens heuristically without double-counting them into billable tokens.
        """
        raw_bytes = len(result_text.encode("utf-8"))
        est_tokens = estimate_tokens(result_text)

        self.estimated_tool_result_tokens += est_tokens

        record = ToolExecutionRecord(
            step=step,
            action=action,
            arguments=arguments,
            result_bytes=raw_bytes,
            estimated_tool_result_tokens=est_tokens,
            duration_seconds=round(duration_seconds, 5),
            cost_units=cost_units,
            is_parallel=is_parallel,
        )
        self.tool_executions.append(record)
        return record

    @property
    def estimated_cost_usd(self) -> float | None:
        """Calculates estimated cost based on pricing config, or None if unknown."""
        if self.pricing is None:
            return None
        return self.pricing.calculate_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            reasoning_included_in_output=not self._any_reasoning_separate,
        )

    def get_metrics_summary(self) -> dict[str, Any]:
        """Returns a clean, decoupled summary of all 8 resource metrics."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "tool_result_tokens": self.estimated_tool_result_tokens,
            "tool_result_tokens_is_estimate": True,
            "tool_calls": len(self.tool_executions),
            "api_calls": self.api_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "wall_clock_seconds": self.wall_clock_seconds,
            "tool_executions": [
                {
                    "step": rec.step,
                    "action": rec.action,
                    "arguments": rec.arguments,
                    "result_bytes": rec.result_bytes,
                    "estimated_tool_result_tokens": rec.estimated_tool_result_tokens,
                    "duration_seconds": rec.duration_seconds,
                    "cost_units": rec.cost_units,
                    "is_parallel": rec.is_parallel,
                }
                for rec in self.tool_executions
            ],
        }
