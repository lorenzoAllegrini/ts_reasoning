"""Groq vision-LM backend for the labelling descriptor — describes PLOTS of windows.

`GroqVLM.describe(system, user, png)` sends one rendered image (≤4 subplots, from
plots.render_batch_png) plus the stage prompt to a Groq-hosted multimodal model and
returns the text. Deterministic (temperature 0, reasoning off for qwen); `<think>`
blocks are stripped defensively. Cross-key failover mirrors the loop's LLM policy:
a per-DAY cap abandons the key (GROQ_API_KEY → GROQ_API_KEY2 → …), a per-MINUTE
throttle backs off and retries the same key.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any

from src import config
from src.agentic.agents.base import _rate_limit_kind, _retry_after_seconds

logger = logging.getLogger(__name__)

_KEY_ENV_VARS = ("GROQ_API_KEY", "GROQ_API_KEY2", "GROQ_API_KEY3")
_MAX_BACKOFF_S = 20.0
_ATTEMPTS_PER_KEY = 3
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class GroqVLM:
    """The one capability the VLM descriptor needs: (system, user, png) → text."""

    def __init__(
        self,
        model: str = config.GROQ_VLM_MODEL,
        max_tokens: int = config.VLM_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._keys = [os.environ[name] for name in _KEY_ENV_VARS if os.environ.get(name)]
        if not self._keys:
            self._keys = [os.environ["GROQ_API_KEY"]]  # preserve the hard failure when unset

    def describe(self, system: str, user: str, png: bytes) -> str:
        from groq import Groq  # noqa: PLC0415 — import at call time keeps module import light

        b64 = base64.b64encode(png).decode()
        messages: list[Any] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]
        last_exc: Exception | None = None
        for key_idx, key in enumerate(self._keys):
            client = Groq(api_key=key)
            for attempt in range(1, _ATTEMPTS_PER_KEY + 1):
                try:
                    if "qwen" in self.model:  # thinking model → answer directly, no <think> budget
                        r = client.chat.completions.create(
                            model=self.model, messages=messages, temperature=0,
                            max_tokens=self.max_tokens, reasoning_effort="none",
                        )
                    else:
                        r = client.chat.completions.create(
                            model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens,
                        )
                    text = r.choices[0].message.content or ""
                    return _THINK_RE.sub("", text).strip()
                except Exception as exc:  # noqa: BLE001 — same retry/fail-over taxonomy as the loop LLM
                    last_exc = exc
                    kind = _rate_limit_kind(exc)
                    if kind == "daily":
                        logger.warning("VLM key #%d daily-capped → %s", key_idx + 1,
                                       "next key" if key_idx + 1 < len(self._keys) else "no keys left")
                        break
                    if kind == "minute" and attempt < _ATTEMPTS_PER_KEY:
                        wait = min(_retry_after_seconds(exc, default=6.0) + 0.5, _MAX_BACKOFF_S)
                        logger.warning("VLM key #%d throttled → backoff %.1fs", key_idx + 1, wait)
                        time.sleep(wait)
                        continue
                    logger.warning("VLM key #%d attempt %d/%d failed: %s",
                                   key_idx + 1, attempt, _ATTEMPTS_PER_KEY, exc)
        raise RuntimeError(f"VLM describe failed after all keys/attempts: {last_exc}")
