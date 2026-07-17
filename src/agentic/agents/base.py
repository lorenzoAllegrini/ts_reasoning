"""BaseLLMAgent — shared LLM plumbing for the three reasoning agents.

Each agent (Analysis, Orchestrator, Perceptor) is a class that owns an LLM and a
system prompt and exposes domain methods; the graph's node methods translate
between the LangGraph state and these methods. Structured output goes through
`json_schema` with retry+validation (§0.4). Only agents use an LLM — the
deterministic core (tools, adjudication, loop, nodes) never does (INV-1).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src import config

logger = logging.getLogger(__name__)

_MAX_BACKOFF_S = 20.0  # never block longer than this on a transient throttle


def _rate_limit_kind(exc: Exception) -> str | None:
    """Classify a 429. 'daily' = a per-DAY cap (TPD/RPD): the key is spent for the day,
    so fail over to the next key. 'minute' = a per-MINUTE throttle (TPM/RPM): transient,
    resets in seconds, so back off briefly and retry the SAME key. None = not a rate limit."""
    s = str(exc).lower()
    if "429" not in s and "rate_limit" not in s and "rate limit" not in s:
        return None
    if "per day" in s or "(tpd)" in s or "(rpd)" in s:
        return "daily"
    return "minute"  # per-minute or any other rate limit → treat as transient & retriable


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """Parse Groq's 'try again in 5.1375s' / '1m2s' hint; fall back to `default`."""
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(exc))
    if not m:
        return default
    return float(m.group(1) or 0) * 60 + float(m.group(2))


class BaseLLMAgent:
    """Owns an LLM (and optional fallback keys) and provides structured / free-text
    invocation helpers.

    `llm` may be None for SCRIPTED subclasses that override every LLM-using method
    (the deterministic worked example / topology stubs); such subclasses never call
    structured()/text(), which assert an LLM is present. `fallbacks` are additional
    equivalent models on different API keys: a rate-limit on one transparently
    advances to the next (see get_llms()).
    """

    def __init__(
        self, llm: BaseChatModel | None = None, fallbacks: Sequence[BaseChatModel] | None = None
    ) -> None:
        self.llm = llm  # the primary (kept for backward compat / introspection)
        self._models: list[BaseChatModel] = [m for m in (llm, *(fallbacks or ())) if m is not None]

    def _invoke_failover(
        self, make_runnable: Callable[[BaseChatModel], Runnable[Any, Any]], messages: list[Any], what: str
    ) -> Any:
        """Invoke across keys robustly. A per-DAY cap abandons the key and fails over;
        a per-MINUTE throttle backs off and retries the SAME key; a validation/transient
        error retries the same key. Raise only when every key/attempt is exhausted."""
        assert self._models, f"{what} requires an LLM"
        retries = config.STRUCTURED_OUTPUT_MAX_RETRIES
        last_exc: Exception | None = None
        for key_idx, model in enumerate(self._models):
            runnable = make_runnable(model)
            for attempt in range(1, retries + 1):
                try:
                    result = runnable.invoke(messages)
                    if result is None:
                        raise ValueError("LLM returned None")
                    if key_idx or attempt > 1:
                        logger.info("%s succeeded on key #%d (attempt %d)", what, key_idx + 1, attempt)
                    return result
                except Exception as exc:  # noqa: BLE001 — retry/fail-over on any provider/validation error
                    last_exc = exc
                    kind = _rate_limit_kind(exc)
                    if kind == "daily":  # key spent for the day → straight to the next key
                        has_next = key_idx + 1 < len(self._models)
                        logger.warning("%s key #%d daily-capped → %s", what, key_idx + 1,
                                       "next key" if has_next else "no keys left")
                        break
                    if kind == "minute" and attempt < retries:  # transient throttle → wait, same key
                        wait = min(_retry_after_seconds(exc, default=6.0) + 0.5, _MAX_BACKOFF_S)
                        logger.warning("%s key #%d throttled (per-minute) → backoff %.1fs, retry same key",
                                       what, key_idx + 1, wait)
                        time.sleep(wait)
                        continue
                    logger.warning("%s key #%d attempt %d/%d failed: %s", what, key_idx + 1, attempt, retries, exc)
        raise RuntimeError(f"{what} failed after all keys/attempts: {last_exc}")

    def structured(self, schema: type[BaseModel], system: str, human: str) -> BaseModel:
        """Call for structured output, retrying on validation failure and failing over
        across keys. On the final failure, raise (never silently loosen — §11.3)."""
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        result = self._invoke_failover(
            lambda m: m.with_structured_output(schema, method=config.STRUCTURED_OUTPUT_METHOD),
            messages, f"structured[{schema.__name__}]",
        )
        return result  # type: ignore[no-any-return]

    def text(self, system: str, human: str) -> str:
        """Call for a plain free-text response (with the same cross-key failover)."""
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        msg = self._invoke_failover(lambda m: m, messages, "text")
        return msg.content if isinstance(msg.content, str) else str(msg.content)
