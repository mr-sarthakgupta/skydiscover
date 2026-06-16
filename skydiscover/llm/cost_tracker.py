"""LLM cost tracker for SkyDiscover — tracks token usage and enforces budget limits."""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("skydiscover.cost")


class CostLimitExceeded(RuntimeError):
    """Raised when cumulative LLM spend reaches the configured budget."""


# Claude Sonnet 4.6 on AWS Bedrock (per-token pricing)
_BEDROCK_SONNET_46_PRICING = {
    "input": 3.0 / 1_000_000,
    "output": 15.0 / 1_000_000,
    "cache_write_5m": 3.75 / 1_000_000,
    "cache_write_1h": 6.0 / 1_000_000,
    "cache_read": 0.30 / 1_000_000,
}

# Fallback for unknown models — same as Sonnet 4.6 to be conservative
_DEFAULT_PRICING = _BEDROCK_SONNET_46_PRICING

_MODEL_PRICING: dict[str, dict[str, float]] = {
    "us.anthropic.claude-sonnet-4-6": _BEDROCK_SONNET_46_PRICING,
    "anthropic.claude-sonnet-4-6": _BEDROCK_SONNET_46_PRICING,
    "us.anthropic.claude-sonnet-4-5-v2": _BEDROCK_SONNET_46_PRICING,
}


def _get_pricing(model: str) -> dict[str, float]:
    for key, pricing in _MODEL_PRICING.items():
        if key in model:
            return pricing
    return _DEFAULT_PRICING


def _cache_write_price(pricing: dict[str, float]) -> float:
    ttl = os.environ.get("BEDROCK_PROMPT_CACHE_TTL", "1h").strip().lower()
    if ttl == "1h":
        return pricing.get("cache_write_1h", 6.0 / 1_000_000)
    return pricing.get("cache_write_5m", 3.75 / 1_000_000)


class CostTracker:
    """Thread-safe cumulative LLM cost tracker with budget enforcement."""

    def __init__(self, max_cost: float = float("inf")):
        self.max_cost = max_cost
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_cost = 0.0
        self.total_calls = 0

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model: str = "",
    ) -> float:
        """Record token usage from one LLM call. Returns the cost of this call."""
        pricing = _get_pricing(model)

        regular_input = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
        call_cost = (
            regular_input * pricing["input"]
            + output_tokens * pricing["output"]
            + cache_read_tokens * pricing["cache_read"]
            + cache_write_tokens * _cache_write_price(pricing)
        )

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cache_read_tokens += cache_read_tokens
            self.total_cache_write_tokens += cache_write_tokens
            self.total_cost += call_cost
            self.total_calls += 1
            snapshot_cost = self.total_cost
            snapshot_calls = self.total_calls

        logger.info(
            "LLM call #%d: in=%d out=%d cache_r=%d cache_w=%d "
            "call=$%.4f total=$%.4f / $%.2f",
            snapshot_calls,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            call_cost,
            snapshot_cost,
            self.max_cost,
        )
        return call_cost

    def check_budget(self) -> None:
        """Raise CostLimitExceeded if cumulative cost meets or exceeds the budget."""
        with self._lock:
            if self.total_cost >= self.max_cost:
                raise CostLimitExceeded(
                    f"Cost limit reached: ${self.total_cost:.4f} >= "
                    f"${self.max_cost:.2f} after {self.total_calls} LLM calls"
                )

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        elapsed = time.monotonic() - self._start_time
        with self._lock:
            remaining = max(0.0, self.max_cost - self.total_cost)
            return (
                f"{'=' * 50}\n"
                f"  LLM Cost Summary\n"
                f"{'=' * 50}\n"
                f"  Total API calls:      {self.total_calls}\n"
                f"  Input tokens:         {self.total_input_tokens:,}\n"
                f"  Output tokens:        {self.total_output_tokens:,}\n"
                f"  Cache read tokens:    {self.total_cache_read_tokens:,}\n"
                f"  Cache write tokens:   {self.total_cache_write_tokens:,}\n"
                f"  Total cost:           ${self.total_cost:.4f}\n"
                f"  Budget:               ${self.max_cost:.2f}\n"
                f"  Remaining:            ${remaining:.4f}\n"
                f"  Elapsed:              {elapsed:.1f}s\n"
                f"{'=' * 50}"
            )


# Module-level singleton — importable from anywhere in the process.
global_cost_tracker = CostTracker()
