"""Base agent class for all MatResearcher agents."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

from rich.console import Console
from rich.markup import escape
import yaml

from ..tools.llm import LLMClient
from ..state import WorkflowState

console = Console()

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"

# Strip Rich markup tags for plain-text log output
_RICH_TAG_RE = re.compile(r"\[/?\w+\]")


def _strip_rich_markup(text: str) -> str:
    return _RICH_TAG_RE.sub("", text)


class BaseAgent:
    """Abstract base for all agents in the MatResearcher pipeline.

    Each agent:
    - Has a name and role description
    - Has access to shared tools (LLM, Sciverse, MinerU, etc.)
    - Has a prompt template loaded from config/prompts/
    - Implements run() which takes WorkflowState and returns partial state updates
    - Writes logs to both console (Rich) and a file under log_dir
    """

    name: str = "base"
    role: str = "Base Agent"
    prompt_file: str = ""  # filename in config/prompts/

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        config: dict | None = None,
        log_dir: str | None = None,
    ):
        self.llm = llm
        self.config = config or {}
        self._prompt: str | None = None
        self._log_file: TextIO | None = None
        self._log_path: str | None = None

        # Init file logging if a log_dir is provided
        if log_dir:
            self._init_logging(log_dir)

    def _init_logging(self, log_dir: str):
        """Open a timestamped log file under log_dir."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        filename = log_path / f"{self.name}.log"
        try:
            self._log_file = open(str(filename), "w", encoding="utf-8")
            self._log_path = str(filename)
            self._log_file.write(f"# Log started: {datetime.now().isoformat()}\n")
            self._log_file.write(f"# Agent: {self.name} ({self.role})\n\n")
            self._log_file.flush()
        except OSError:
            self._log_file = None

    def close_log(self):
        """Close the log file if open."""
        if self._log_file:
            self._log_file.write(f"\n# Log ended: {datetime.now().isoformat()}\n")
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        """Ensure log file is closed on garbage collection (crash safety).

        Guard with getattr: if __init__ raised part-way (or the object was
        created via __new__ without __init__, as replay/debug scripts do),
        `_log_file` is never set — accessing it directly raises AttributeError
        at GC time and masks the original error.
        """
        log_file = getattr(self, "_log_file", None)
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
            self._log_file = None

    @property
    def log_path(self) -> str | None:
        """Return the log file path, if logging is active."""
        return self._log_path

    @property
    def prompt(self) -> str:
        """Load and cache the prompt template."""
        if self._prompt is None:
            if self.prompt_file:
                path = PROMPTS_DIR / self.prompt_file
                if path.exists():
                    self._prompt = path.read_text(encoding="utf-8")
                else:
                    self._prompt = ""
            else:
                self._prompt = ""
        return self._prompt

    async def run(self, state: WorkflowState) -> dict:
        """Execute the agent's task.

        Args:
            state: Current WorkflowState dict.

        Returns:
            Partial state dict to merge back into WorkflowState.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    def log(self, msg: str, style: str = "green"):
        """Log a message to console and to the log file (if configured)."""
        # Console output with Rich styling.
        # style=None → omit the style tag entirely (otherwise `[None]`/`[/None]`
        # is emitted and Rich raises MarkupError on the closing tag).
        # Fallback: if the message contains stray '[' / ']' (literature titles,
        # error text, LLM output), Rich markup parsing can crash the pipeline —
        # re-print with markup escaped so logging can never throw.
        label = f"[{self.name}] " if not style else f"[{style}][{self.name}][/{style}] "
        try:
            console.print(label + msg)
        except Exception:
            console.print(escape(label + msg))

        # File output (plain text, no markup)
        if self._log_file:
            ts = datetime.now().strftime("%H:%M:%S")
            plain = _strip_rich_markup(msg)
            self._log_file.write(f"[{ts}] [{self.name}] {plain}\n")
            self._log_file.flush()

    def log_step(self, step: str, title: str, kind: str = "start"):
        """Print a green step banner for sub-step visibility within multi-step agents.

        Uses the same green (▸/✔) style as nodes.py step banners
        so sub-step banners are visually consistent with pipeline-level step banners.
        """
        if kind == "start":
            console.print(f"\n  [bold green]▸ Step {step}:[/bold green] {title}")
        else:
            console.print(f"  [bold green]✔ Step {step} 完成:[/bold green] {title}")

        # File output (plain text, no markup)
        if self._log_file:
            ts = datetime.now().strftime("%H:%M:%S")
            label = "START" if kind == "start" else "DONE"
            plain = _strip_rich_markup(f"Step {step} {label}: {title}")
            self._log_file.write(f"[{ts}] [{self.name}] {plain}\n")
            self._log_file.flush()

    def load_config(self, config_file: str) -> dict:
        """Load a YAML config file."""
        path = Path(__file__).resolve().parents[2] / "config" / config_file
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
