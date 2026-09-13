"""LLM client wrapper (OpenAI-compatible API).

Supports OpenAI, Qwen2.5 (via vLLM/Ollama), and any OpenAI-compatible endpoint.
Embedding uses local HuggingFace models (free, no API cost).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from ..env_loader import load_env_files
from .token_counter import TokenCounter, estimate_tokens

load_env_files()


def _is_empty_response(exc: BaseException) -> bool:
    """Classify the empty-response ValueError (for diagnostics).

    NOTE (2026-08-14 replay experiment): this failure is NOT deterministic.
    The exact same chunks failed 0/3 when replayed back-to-back, then
    succeeded 6/6 with 8 s spacing (finish_reason=stop, valid JSON).  It is
    a TRANSIENT provider-side event — callers must retry with the SAME text,
    never treat it as content rejection."""
    return isinstance(exc, ValueError) and "empty response" in str(exc)


def _should_retry(exc: BaseException) -> bool:
    """Retry policy for ``complete_json`` — retry EVERY exception with the
    same text.

    Previously the empty-response ValueError was exempted from retry on the
    assumption that it was a deterministic content rejection.  The 2026-08-14
    replay experiment disproved this: chunks that failed 0/3 back-to-back
    succeeded 6/6 spaced 8 s apart.  Retrying same-text after exponential
    backoff fixes the failure; the old policy instead made callers SHORTEN
    the chunk, discarding half the content for no reason."""
    return True


# api_base substring → the environment variable holding that provider's key.
_PROVIDER_KEY_VARS: list[tuple[str, str]] = [
    ("minimaxi", "MINIMAX_API_KEY"),
    ("dashscope", "DASHSCOPE_API_KEY"),
    ("aliyuncs", "DASHSCOPE_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("moonshot", "MOONSHOT_API_KEY"),
    ("openai.com", "OPENAI_API_KEY"),
]


def _provider_api_key(api_base: str | None) -> str | None:
    """Pick the API key belonging to the provider implied by ``api_base``.

    Returns None when no provider-specific variable is set — callers must treat
    that as "unconfigured" rather than borrowing another provider's key.
    """
    base = (api_base or "").lower()
    for token, env_var in _PROVIDER_KEY_VARS:
        if token in base:
            # Prefer the provider-specific key, but fall back to the universal
            # LLM_API_KEY so a single secret works for any OpenAI-compatible endpoint.
            return os.getenv(env_var) or os.getenv("LLM_API_KEY")
    # Unknown/local endpoint (vLLM, Ollama, ...): accept the generic key.
    return os.getenv("LLM_API_KEY")


class LLMClient:
    """Async LLM client with structured output support and local embeddings."""

    # Default API timeout (seconds) — prevents hanging on slow/unreachable endpoints
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        timeout: float | None = None,
        token_counter: TokenCounter | None = None,
        max_retries: int = 0,
    ):
        self.api_base = api_base or os.getenv("LLM_API_BASE", "https://llm-svkdmm9vxy5rc316.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        # Key resolution order matches nodes.py: the explicit env override,
        # then the active provider key (MiniMax since 2026-08-14), then the
        # legacy DashScope key for backward compatibility with old configs.
        # Without MINIMAX_API_KEY here, parameterless LLMClient() silently
        # authenticated with a stale DASHSCOPE_API_KEY → 401 on api.minimaxi.com.
        # Provider-aware key resolution. The old chain fell through to
        # DASHSCOPE_API_KEY regardless of api_base, so a run against
        # api.minimaxi.com authenticated with a stale DashScope key and failed
        # with an opaque 401 several minutes in. Now the key must match the
        # provider implied by api_base (see config_check.py).
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or _provider_api_key(self.api_base)
        )
        self.model = model or os.getenv("LLM_MODEL", "qwen3.7-max-2026-05-17")

        # Fail fast (and loudly) on the misconfigurations that used to surface
        # as HTTP 401 mid-run.
        from ..config_check import validate_runtime_config

        for problem in validate_runtime_config(
            api_base=self.api_base, api_key=self.api_key, model=self.model
        ):
            if "missing" in problem or "truncated" in problem:
                raise RuntimeError(
                    f"{problem}\n"
                    f"Resolved target: api_base={self.api_base} model={self.model}"
                )
            print(f"[config] WARNING: {problem}")
        self.temperature = temperature
        self._timeout = timeout or self.DEFAULT_TIMEOUT
        self._token_counter = token_counter
        # SDK-internal retries are DISABLED by default: the OpenAI client's
        # max_retries=2 amplifies timeouts into multi-minute stalls (e.g.
        # submission generation 2026-08-14: 3 x 120 s = 362 s before failing).
        # Retries are owned by the layers that know the failure semantics:
        #   - complete_json  -> tenacity (3 attempts, exponential backoff)
        #   - complete()     -> caller try/except fallback (report/submission)
        # Callers may opt back in by passing max_retries > 0.
        self._client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=httpx.Timeout(self._timeout, connect=30.0),
            max_retries=max_retries,
        )
        self._embedding_model = None  # lazy-loaded HuggingFace model

    async def complete(
        self, system: str, user: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        stage: str | None = None,
        timeout: float | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Get a chat completion. (Retry handled by caller complete_json.)

        Args:
            stage: Pipeline stage label for token accounting (usually the
                calling agent's `name`). If a TokenCounter is attached, usage
                is recorded after a successful call and errors are counted.
            timeout: Per-request timeout in seconds, overriding the client
                default. Long generations (max_tokens >= 16384 on reasoning
                models) routinely exceed the 120 s default — pass a generous
                value (300-600) for report/submission-style tasks.
            disable_thinking: MiniMax-only. When True, sends
                ``{"thinking": {"type": "disabled"}}`` so the model skips the
                <think> chain entirely — right choice for template-driven
                reformatting tasks (competition submission) where the raw
                survey report already contains the analysis. Content comes
                back faster and the whole max_tokens budget goes to the
                document. When False (default), keeps reasoning_split=True so
                thinking lands in reasoning_content, not content.
        """
        try:
            request: dict = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens,
            )
            # MiniMax-M3 (2026-08-14 gap failure): thinking is ON by default and,
            # without reasoning_split, the <think> chain is emitted INSIDE
            # content — corrupting JSON parsing and silently burning max_tokens
            # budget (4.5-min call, 3x ValueError, rule-based fallback).
            # reasoning_split moves thinking to reasoning_content so that
            # content stays clean for complete_json. Non-MiniMax endpoints are
            # unaffected (extra_body is only added for MiniMax models).
            if "minimax" in self.model.lower():
                if disable_thinking:
                    request["extra_body"] = {"thinking": {"type": "disabled"}}
                else:
                    request["extra_body"] = {"reasoning_split": True}
            # Per-request timeout override (long document generations).
            if timeout is not None:
                request["timeout"] = timeout
            resp = await self._client.chat.completions.create(**request)
        except Exception as e:
            # Failed calls still burn quota (esp. retry loops) — count them.
            if self._token_counter is not None:
                self._token_counter.record_error(stage)
            raise
        content = resp.choices[0].message.content or ""
        self._record_usage(stage, resp, system, user, content)
        return content

    def _record_usage(
        self,
        stage: str | None,
        resp,
        system: str,
        user: str,
        content: str,
    ):
        """Record token usage into the shared TokenCounter.

        Prefers the API `usage` object (exact); falls back to a character-based
        estimate when the endpoint omits it.
        """
        if self._token_counter is None:
            return
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is not None and completion_tokens is not None:
            self._token_counter.record(
                stage, prompt_tokens, completion_tokens, estimated=False,
            )
        else:
            pt = estimate_tokens(system) + estimate_tokens(user)
            ct = estimate_tokens(content)
            self._token_counter.record(stage, pt, ct, estimated=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=3, max=30),
        # Retry every failure — including the empty-response ValueError —
        # with the SAME text and exponential backoff (3s → 6s waits over 3
        # attempts).  Empty responses are TRANSIENT provider-side events
        # (2026-08-14 replay: 0/3 back-to-back vs 6/6 with 8 s spacing), so
        # same-text retries fix them; shortening the prompt only discards
        # content.  knowledge_extraction._extract_chunk unwraps the RetryError
        # and applies its own recovery for whatever survives.
        retry=retry_if_exception(_should_retry),
    )
    async def complete_json(
        self, system: str, user: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
        stage: str | None = None,
        timeout: float | None = None,
        disable_thinking: bool = False,
    ) -> dict | list:
        """Get a chat completion and parse as JSON with robust content extraction.

        Supports truncated-JSON recovery: if output was cut off by max_tokens,
        attempts to close the JSON structure and retry parsing.
        """
        content = await self.complete(
            system, user, temperature,
            max_tokens=max_tokens, stage=stage,
            timeout=timeout, disable_thinking=disable_thinking,
        )
        raw = content.strip()

        if not raw:
            err = ValueError(
                "LLM returned empty response. "
                "The model may have rejected the prompt (content filter) or produced no output."
            )
            err.raw_content = content  # full raw output for diagnostics
            raise err

        # Try to extract JSON block from markdown fences: ```json ... ``` or ``` ... ```
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

        # If still not parseable, try extracting JSON via bracket balancing
        if raw:
            # Try direct JSON parse first
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            # Try bracket-balanced extraction — strip trailing non-JSON text
            # that LLMs sometimes append after the closing brace
            json_text = self._extract_balanced_json(raw)
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            # Fallback: regex-based extraction (handles cases where bracket
            # balancing fails due to braces inside string values)
            for pattern in (r'\{.*\}', r'\[.*\]'):
                m = re.search(pattern, raw, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        continue

        # ── Truncated JSON recovery ──
        # If max_tokens cut off the output, try to salvage by closing the structure.
        recovered = self._recover_truncated_json(raw)
        if recovered is not None:
            return recovered

        err = ValueError(
            f"LLM returned non-JSON content. "
            f"Raw preview ({len(content)} chars): {content[:500]}"
        )
        err.raw_content = content  # full raw output for offline diagnostics
        raise err

    @staticmethod
    def _extract_balanced_json(text: str) -> str | None:
        """Extract a balanced JSON object or array by tracking brace depth.

        Finds the first '{' or '[' and scans forward, counting opening/closing
        braces (ignoring those inside quoted strings), until the depth returns
        to zero.  This correctly handles nested objects and strips any trailing
        commentary that LLMs sometimes append after the JSON.
        """
        text = text.strip()
        if not text:
            return None

        # Find the first JSON opener
        first_brace = -1
        opener = ''
        closer = ''
        for i, ch in enumerate(text):
            if ch == '{':
                first_brace = i
                opener, closer = '{', '}'
                break
            elif ch == '[':
                first_brace = i
                opener, closer = '[', ']'
                break

        if first_brace < 0:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(first_brace, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == '\\' and in_string:
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[first_brace:i + 1]

        # Depth never returned to zero — probably truncated
        return None

    @staticmethod
    def _recover_truncated_json(text: str) -> dict | list | None:
        """Attempt to salvage a JSON string that was truncated by max_tokens.

        Strategy 1: find the LAST complete top-level element and close the
        container there.  This handles deep nesting (e.g. an array of objects
        cut mid-element) where simple trailing-comma heuristics fail — the
        exact shape that broke gap generation on 2026-08-14.
        Strategy 2 (legacy): progressively strip trailing incomplete content
        and try to close the JSON structure with '}' or ']'.
        """
        text = text.strip()
        if not (text.startswith('{') or text.startswith('[')):
            return None

        # ── Strategy 1: furthest complete top-level element ──
        # Track {} and [] depths separately; remember the last index where a
        # complete element ended.  An OBJECT closing at depth 0 is always a
        # valid salvage point — even when the outer array is still open (we
        # simply append the missing ']').  An ARRAY closing at depth 0 is only
        # a point when braces are also closed (complete JSON).
        depth_braces = 0
        depth_brackets = 0
        last_complete = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth_braces += 1
            elif ch == '}':
                depth_braces -= 1
                if depth_braces == 0:
                    last_complete = i
            elif ch == '[':
                depth_brackets += 1
            elif ch == ']':
                depth_brackets -= 1
                if depth_braces == 0 and depth_brackets == 0:
                    last_complete = i
        if last_complete > 0:
            # Close the outer container (if it was left open) and parse.
            outer = text[0]
            closer = '}' if outer == '{' else ']'
            candidate = text[:last_complete + 1].rstrip().rstrip(',') + closer
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # ── Strategy 2 (legacy): strip trailing fragments ──
        for closing in ('}', ']'):
            # Strip the last incomplete fragment (trailing comma, partial key, etc.)
            # Try all reasonable cut points from end backwards; first success wins.
            for cut in (text.rfind(',"'), text.rfind(', "'), text.rfind('\n'),
                        text.rfind('",'), text.rfind('" ')):
                if cut < 0:
                    continue
                candidate = text[:cut].rstrip().rstrip(',') + closing
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        return None

    # ── Embedding (local HuggingFace models — free, no API cost) ──

    def embed(self, text: str) -> list[float]:
        """Embed a single text using a local HuggingFace model.

        Uses the model specified by EMBEDDING_MODEL env var (default: BAAI/bge-m3).
        No API call — runs entirely on the local machine.
        """
        self._load_embedding_model()
        vec = self._embedding_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch."""
        self._load_embedding_model()
        vecs = self._embedding_model.encode(
            texts, normalize_embeddings=True, batch_size=32,
        )
        return vecs.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the loaded model."""
        self._load_embedding_model()
        try:
            return self._embedding_model.get_sentence_embedding_dimension()
        except AttributeError:
            return 1024  # bge-m3 default

    def _load_embedding_model(self):
        """Lazy-load the HuggingFace embedding model on first use."""
        if self._embedding_model is not None:
            return
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
