"""Part 1 — the agentic telemetry-interpretation loop.

A bounded LangGraph agent (orchestrator + statistical analyst + ChatTS perceptor +
mission knowledge graph) that investigates a telemetry window by asking increasingly
targeted questions and concludes (ACCEPT) on its best-supported hypothesis.
Run: `uv run python -m src.agentic.main --stub` (no network) or without --stub (Groq).
Sibling part: `src.labelling` (rule-based OOL detection + multi-granularity descriptions).
"""
