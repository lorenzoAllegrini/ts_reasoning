"""get_llm() — the UNIQUE point where an LLM is instantiated (§0.3).

Changing provider (Groq → Ollama in-perimeter, → Gemini, → other) means changing
ONLY this function. No node instantiates a client directly. This is a requirement,
not a style choice (see §0.3, §10).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src import config

load_dotenv()


# Primary key first, then any fallbacks. gpt-oss-20b's free tier has a 200k tokens/day
# cap; a second key (GROQ_API_KEY2) lets a long run fail over instead of dying at the cap.
_KEY_ENV_VARS = ("GROQ_API_KEY", "GROQ_API_KEY2", "GROQ_API_KEY3")


def _build_groq(api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=SecretStr(api_key),
        temperature=temperature,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def get_llm(temperature: float = config.LLM_TEMPERATURE) -> ChatGroq:
    """Factory for every LLM node (orchestrator, rationale writer) — the PRIMARY key.

    Groq for this validation phase (§0.3, §10 — data is public ESA-ADB + local
    synthetic, so an external API is legitimate). temperature defaults to 0.0 so
    the loop is reproducible.
    """
    return _build_groq(os.environ["GROQ_API_KEY"], temperature)


def get_llms(temperature: float = config.LLM_TEMPERATURE) -> list[ChatGroq]:
    """The primary LLM followed by one per configured fallback key (GROQ_API_KEY2…).

    Agents fail over to the next key when the current one is rate-limited (per-day
    token cap), so a long run survives exhausting a single free-tier key. Returns at
    least the primary; extra keys are included only when their env var is set.
    """
    keys = [os.environ[name] for name in _KEY_ENV_VARS if os.environ.get(name)]
    if not keys:  # preserve the original hard failure when nothing is configured
        keys = [os.environ["GROQ_API_KEY"]]
    return [_build_groq(k, temperature) for k in keys]
