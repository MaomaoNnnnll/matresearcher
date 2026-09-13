"""Startup configuration validation.

Why this module exists
----------------------
Three config surfaces used to drift apart (``workflow.yaml`` pinned
``${MINIMAX_API_KEY}`` while ``.env.example`` and the code default targeted
DashScope/Qwen).  They now share a single canonical secret, ``LLM_API_KEY``
(set in ``~/.matresearcher/secrets.env``), with ``DASHSCOPE_API_KEY`` /
``MINIMAX_API_KEY`` retained only as provider-specific aliases that
``tools/llm.py`` resolves from ``api_base``.  :func:`validate_runtime_config`
turns any remaining mismatch into an immediate, actionable error instead of an
opaque HTTP 401 minutes into a run.

It is deliberately dependency-free so it can run before any client is built.
"""
from __future__ import annotations

import os
import re
from typing import Any

# Substring in api_base → (provider label, accepted key env vars, accepted model hint)
_PROVIDERS: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    ("minimaxi", "MiniMax", ("MINIMAX_API_KEY", "LLM_API_KEY"), ("minimax", "m1", "m2")),
    ("dashscope", "DashScope", ("DASHSCOPE_API_KEY", "LLM_API_KEY"), ("qwen", "qwq")),
    ("aliyuncs", "DashScope", ("DASHSCOPE_API_KEY", "LLM_API_KEY"), ("qwen", "qwq")),
    ("deepseek", "DeepSeek", ("DEEPSEEK_API_KEY", "LLM_API_KEY"), ("deepseek",)),
    ("openai.com", "OpenAI", ("OPENAI_API_KEY", "LLM_API_KEY"), ("gpt", "o1", "o3")),
    ("moonshot", "Moonshot", ("MOONSHOT_API_KEY", "LLM_API_KEY"), ("kimi", "moonshot")),
]

_PLACEHOLDERS = {"", "your-key", "your_api_key", "sk-xxxxx", "changeme", "none", "null"}


def _is_placeholder(value: str | None) -> bool:
    return value is None or value.strip().lower() in _PLACEHOLDERS


def _match_provider(api_base: str | None, model: str | None) -> str | None:
    """Identify the provider implied by api_base, then by model name."""
    haystacks = [(api_base or "").lower(), (model or "").lower()]
    for idx, hay in enumerate(haystacks):
        for token, label, _, model_hints in _PROVIDERS:
            probe = token if idx == 0 else ""
            if idx == 0 and probe and probe in hay:
                return label
            if idx == 1:
                for hint in model_hints:
                    if hint and hay.startswith(hint):
                        return label
    return None


def _accepted_envs(provider: str | None) -> tuple[str, ...]:
    for _token, label, envs, _hints in _PROVIDERS:
        if label == provider:
            return envs
    return ("LLM_API_KEY",)


def validate_runtime_config(
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    require_sciverse: bool = False,
) -> list[str]:
    """Return a list of human-readable configuration problems ([] == healthy)."""
    problems: list[str] = []

    api_base = api_base or os.getenv("LLM_API_BASE")
    model = model or os.getenv("LLM_MODEL")
    api_key = api_key or os.getenv("LLM_API_KEY")
    provider = _match_provider(api_base, model)

    # 1) An API key must exist and must not be a template placeholder.
    if _is_placeholder(api_key):
        envs = " / ".join(_accepted_envs(provider))
        problems.append(
            f"LLM API key is missing. Provider detected: {provider or 'unknown'} "
            f"(api_base={api_base or 'unset'}). Set one of: {envs} — "
            f"recommended location: ~/.matresearcher/secrets.env"
        )
    elif api_key and len(api_key.strip()) < 16:
        problems.append(
            "LLM API key looks truncated (fewer than 16 characters) — "
            "check ~/.matresearcher/secrets.env"
        )

    # 2) Base URL and model must point at the same provider.
    base_provider = _match_provider(api_base, None)
    model_provider = _match_provider(None, model)
    if base_provider and model_provider and base_provider != model_provider:
        problems.append(
            f"Provider mismatch: api_base points at {base_provider} "
            f"({api_base}) but model '{model}' belongs to {model_provider}. "
            f"A request like this returns HTTP 401."
        )

    # 3) The key env var in use must match the provider.
    if provider and not _is_placeholder(api_key):
        accepted = _accepted_envs(provider)
        if not any(os.getenv(env) for env in accepted):
            problems.append(
                f"A key was supplied for provider {provider} but none of the "
                f"expected variables ({' / '.join(accepted)}) is set — the key "
                f"was probably inherited from another provider's variable."
            )

    # 4) Sciverse drives the whole retrieval stage.
    sciverse_key = os.getenv("SCIVERSE_API_KEY")
    if require_sciverse and _is_placeholder(sciverse_key):
        problems.append(
            "SCIVERSE_API_KEY is not set — literature retrieval will return no "
            "results. Set it in ~/.matresearcher/secrets.env."
        )

    return problems


def assert_runtime_config(**kwargs: Any) -> None:
    """Raise RuntimeError listing every configuration problem found."""
    problems = validate_runtime_config(**kwargs)
    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(f"Invalid runtime configuration:\n  - {joined}")


def describe_runtime_config() -> str:
    """One-line summary of the resolved LLM target (safe: masks the key)."""
    api_base = os.getenv("LLM_API_BASE", "unset")
    model = os.getenv("LLM_MODEL", "unset")
    key = os.getenv("LLM_API_KEY") or ""
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else ("<unset>" if not key else "<short>")
    return f"api_base={api_base} model={model} api_key={masked}"
