"""OpenTelemetry tracing (GenAI semantic conventions) via OpenLLMetry (traceloop-sdk).

Opt-in and LOCAL-FIRST. `init_tracing()` is a no-op unless enabled (env `ESA_TRACING=1`
or the CLI `--trace` flag), so tests and normal runs are untouched. When enabled it
initialises Traceloop, which auto-instruments LangChain/ChatGroq (and, if present, the
ChatTS `transformers` calls): every LLM request becomes an OpenTelemetry span carrying
the GenAI conventions (`gen_ai.request.model`, `gen_ai.usage.*tokens`,
`gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`), and
each LangGraph node becomes an `execute_task <node>` span.

Where the spans go (`_export_target`):
  · `ESA_OTEL_ENDPOINT` set → OTLP to that local collector (e.g. Jaeger at
    `http://localhost:4318`); needs the collector running (Docker).
  · otherwise              → a JSONL FILE, `ESA_OTEL_FILE` or the fixed default
    `traces/traces.jsonl` (one span per line, appended; no infra, no egress).

It NEVER defaults to a remote SaaS, and Traceloop's own anonymous usage telemetry is
disabled. Import-safe even if traceloop-sdk is absent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

_APP_NAME = "esa-telemetry-loop"
DEFAULT_TRACE_FILE = "traces/traces.jsonl"  # fixed sink when no OTLP endpoint is configured

_initialised = False
_trace_file: IO[str] | None = None  # kept alive for the process so the exporter can write


def tracing_enabled() -> bool:
    """True when ESA_TRACING is set to a truthy value."""
    return os.getenv("ESA_TRACING", "0").strip().lower() in ("1", "true", "yes", "on")


def _export_target() -> tuple[str, str]:
    """Resolve where spans go, without side effects (pure — unit-testable).

    Returns ("otlp", endpoint) if ESA_OTEL_ENDPOINT is set, else ("file", path) where
    path is ESA_OTEL_FILE or the fixed default DEFAULT_TRACE_FILE.
    """
    endpoint = os.getenv("ESA_OTEL_ENDPOINT", "").strip()
    if endpoint:
        return ("otlp", endpoint)
    return ("file", os.getenv("ESA_OTEL_FILE", "").strip() or DEFAULT_TRACE_FILE)


def init_tracing(force: bool = False) -> bool:
    """Initialise OTel/OpenLLMetry tracing exactly once. Returns True iff newly initialised.

    A no-op (returns False) unless `force` or `ESA_TRACING` is set. Safe to call from every
    entrypoint; the second call is idempotent.
    """
    global _initialised, _trace_file
    if _initialised or not (force or tracing_enabled()):
        return False
    os.environ.setdefault("TRACELOOP_TELEMETRY", "false")  # no anonymous usage pings from Traceloop
    try:
        from traceloop.sdk import Traceloop
    except ImportError:
        logger.warning("tracing requested but traceloop-sdk is not installed; continuing without tracing")
        return False

    kind, target = _export_target()
    if kind == "otlp":  # ship to a local OTLP collector / Jaeger — no egress beyond your machine
        Traceloop.init(app_name=_APP_NAME, api_endpoint=target, disable_batch=True)
        logger.info("OTel tracing enabled → OTLP %s", target)
    else:  # persist spans to a fixed JSONL file (one span per line, appended)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        _trace_file = path.open("a", buffering=1, encoding="utf-8")  # line-buffered → flushes per span
        exporter = ConsoleSpanExporter(out=_trace_file, formatter=lambda s: s.to_json(indent=None) + "\n")
        Traceloop.init(app_name=_APP_NAME, exporter=exporter, disable_batch=True)
        logger.info("OTel tracing enabled → file %s", path.resolve())
    _initialised = True
    return True
