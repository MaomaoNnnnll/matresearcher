"""Per-stage LLM token accounting and cost estimation.

MatResearcher runs many LLM calls spread across pipeline stages (task_planning,
literature_search, llm_prefilter, ... report_generation, fact_check). This
module gives a single place to accumulate token usage per stage so that:

- Each run reports exactly how many tokens were consumed by which stage
  (cached/resumed nodes never call the LLM, so the counter only reflects
  tokens actually spent in the current process).
- Cost can be estimated when per-million-token prices are configured.

Prices are read from environment variables (optional):

    LLM_PRICE_INPUT_PER_M   — CNY per 1M input tokens   (e.g. "4" = 4 元/百万)
    LLM_PRICE_OUTPUT_PER_M  — CNY per 1M output tokens  (e.g. "12" = 12 元/百万)

If either is unset, the cost column shows "—" (token counts are still tracked).

Thread safety: `record()`/`record_error()` are synchronous and never await, so
within a single asyncio event loop they cannot interleave. A threading.Lock is
still used as cheap defense for exotic embedding/multi-thread usage.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from rich.console import Console
from rich.table import Table


def estimate_tokens(text: str) -> int:
    """Rough character-based token estimate (used when the API omits `usage`).

    Rule of thumb: English ≈ 4 chars/token, CJK ≈ 1.5 chars/token.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / 4.0 + non_ascii / 1.5) + 1


class TokenCounter:
    """Accumulate LLM token usage per pipeline stage."""

    # Optional per-million-token prices (元/百万 tokens)
    PRICE_INPUT_PER_M: Optional[float] = None
    PRICE_OUTPUT_PER_M: Optional[float] = None

    def __init__(self):
        self._lock = threading.Lock()
        # stage -> {"calls": int, "errors": int, "prompt_tokens": int,
        #           "completion_tokens": int, "estimated_calls": int}
        self._stages: dict[str, dict] = {}

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated: bool = False,
    ):
        """Record one LLM call's usage under a stage.

        Args:
            stage: Pipeline stage label (usually the agent's `name`).
            prompt_tokens: Input tokens (from API usage, or estimated).
            completion_tokens: Output tokens (from API usage, or estimated).
            estimated: True when tokens were estimated from character counts
                because the API response carried no `usage`.
        """
        stage = (stage or "unknown").strip() or "unknown"
        pt = max(0, int(prompt_tokens or 0))
        ct = max(0, int(completion_tokens or 0))
        with self._lock:
            bucket = self._stages.setdefault(stage, self._new_bucket())
            bucket["calls"] += 1
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            if estimated:
                bucket["estimated_calls"] += 1

    def record_error(self, stage: str):
        """Record a failed LLM call (still burns quota on retry loops)."""
        stage = (stage or "unknown").strip() or "unknown"
        with self._lock:
            bucket = self._stages.setdefault(stage, self._new_bucket())
            bucket["errors"] += 1

    @staticmethod
    def _new_bucket() -> dict:
        return {
            "calls": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_calls": 0,
        }

    # ── Aggregation ──────────────────────────────────────────────────────

    def summary(self) -> list[dict]:
        """Per-stage rows sorted by total tokens (desc)."""
        rows = []
        with self._lock:
            for stage, b in self._stages.items():
                total = b["prompt_tokens"] + b["completion_tokens"]
                rows.append({
                    "stage": stage,
                    "calls": b["calls"],
                    "errors": b["errors"],
                    "prompt_tokens": b["prompt_tokens"],
                    "completion_tokens": b["completion_tokens"],
                    "total_tokens": total,
                    "estimated_calls": b["estimated_calls"],
                })
        rows.sort(key=lambda r: r["total_tokens"], reverse=True)
        return rows

    def totals(self) -> dict:
        """Aggregate across all stages."""
        t = {"calls": 0, "errors": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0, "estimated_calls": 0}
        for r in self.summary():
            for k in ("calls", "errors", "prompt_tokens", "completion_tokens",
                      "total_tokens", "estimated_calls"):
                t[k] += r[k]
        return t

    # ── Cost ─────────────────────────────────────────────────────────────

    @classmethod
    def _prices(cls) -> tuple[Optional[float], Optional[float]]:
        """Return (input_per_m, output_per_m), reading env overrides each call."""
        if cls.PRICE_INPUT_PER_M is None:
            try:
                cls.PRICE_INPUT_PER_M = float(os.getenv("LLM_PRICE_INPUT_PER_M", ""))
            except ValueError:
                cls.PRICE_INPUT_PER_M = None
        if cls.PRICE_OUTPUT_PER_M is None:
            try:
                cls.PRICE_OUTPUT_PER_M = float(os.getenv("LLM_PRICE_OUTPUT_PER_M", ""))
            except ValueError:
                cls.PRICE_OUTPUT_PER_M = None
        return cls.PRICE_INPUT_PER_M, cls.PRICE_OUTPUT_PER_M

    def stage_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Cost in CNY for a token pair, or None if prices are unconfigured."""
        pin, pout = self._prices()
        if pin is None or pout is None:
            return None
        return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000.0

    # ── Rendering ────────────────────────────────────────────────────────

    def format_table(self, title: str = "Token Usage (本次运行)") -> str:
        """Render the usage table as a string (safe for console / log file)."""
        rows = self.summary()
        totals = self.totals()

        has_price = all(p is not None for p in self._prices())

        table = Table(title=title, title_style="bold cyan")
        table.add_column("环节", style="cyan")
        table.add_column("调用", justify="right")
        table.add_column("失败", justify="right", style="red")
        table.add_column("输入 tok", justify="right")
        table.add_column("输出 tok", justify="right")
        table.add_column("合计 tok", justify="right", style="bold")
        table.add_column("估算", justify="right", style="yellow")
        if has_price:
            table.add_column("费用(¥)", justify="right")

        for r in rows:
            cost = self.stage_cost(r["prompt_tokens"], r["completion_tokens"])
            row = [
                r["stage"],
                str(r["calls"]),
                str(r["errors"]),
                f"{r['prompt_tokens']:,}",
                f"{r['completion_tokens']:,}",
                f"{r['total_tokens']:,}",
                str(r["estimated_calls"]),
            ]
            if has_price:
                row.append(f"{cost:.4f}" if cost is not None else "—")
            table.add_row(*row)

        total_cost = self.stage_cost(totals["prompt_tokens"], totals["completion_tokens"])
        total_row = [
            "TOTAL",
            str(totals["calls"]),
            str(totals["errors"]),
            f"{totals['prompt_tokens']:,}",
            f"{totals['completion_tokens']:,}",
            f"{totals['total_tokens']:,}",
            str(totals["estimated_calls"]),
        ]
        if has_price:
            total_row.append(f"{total_cost:.4f}" if total_cost is not None else "—")
        table.add_row(*total_row, style="bold")

        # If prices are unconfigured, add a hint line
        console = Console()
        buf = []
        with console.capture() as cap:
            console.print(table)
        buf.append(cap.get())

        if not has_price:
            buf.append(
                "  (费用未估算：可设置 LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M "
                "环境变量，单位 元/百万 token)"
            )
        if totals["estimated_calls"]:
            buf.append(
                "  (部分调用未返回 usage，token 数按字符估算，见「估算」列)"
            )
        if totals["errors"]:
            buf.append(
                f"  (⚠ {totals['errors']} 次失败调用同样消耗配额/计费，见「失败」列)"
            )
        return "\n".join(buf)

    def to_dict(self) -> dict:
        """Machine-readable dump (used for token_usage.json)."""
        return {
            "stages": self.summary(),
            "totals": self.totals(),
            "prices": {"input_per_m": self._prices()[0], "output_per_m": self._prices()[1]},
        }
