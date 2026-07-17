# ts_reasoning

Reasoning and labelling over spacecraft telemetry (ESA-ADB), in two sibling parts:

## `src/agentic/` — continuous telemetry interpretation
A bounded LangGraph agent: an orchestrator commands a statistical analyst
(deterministic tools: PELT segmentation, interval statistics, out-of-limits checks,
cross-channel coupling), a ChatTS perceptor (native time-series LLM) and a mission
knowledge graph, asks increasingly targeted questions at shifting granularities,
refines competing hypotheses, and concludes (ACCEPT) on the best-supported one.
Multi-key Groq failover (per-day cap → next key, per-minute throttle → backoff) and
opt-in OpenTelemetry/OpenLLMetry tracing (`ESA_TRACING=1`, spans to
`traces/traces.jsonl`).

```bash
uv run python -m src.agentic.main --stub    # deterministic worked example, no network
uv run python -m src.agentic.main           # real LLM agents (GROQ_API_KEY in .env)
```

## `src/labelling/` — timeseries labelling
Rule-based OOL (out-of-limits) detection over the ESA training telemetry — rolling
median / variance / zero-crossing-rate features, per-channel robust-z thresholds,
selection balanced per subsystem (≤ 200 distinct points, ≥ 1 day apart) — then a
`TimeSeriesDescriptor` describes each point at three independent granularities
(coarse ± 512 / medium ± 128 / fine ± 24 samples) over the most active channels of the
detection channel's subsystem. Two interchangeable backbones: a Groq vision LM on
rendered plots (default; ≤ 4 subplots per call, sequential batches) or local ChatTS on
raw values. Outputs land in `labelling_out/` (CSV + per-point PNGs; resumable).

```bash
uv run python -m src.labelling.run_ool         # detect OOL points (needs the ESA dataset)
uv run python -m src.labelling.run_descriptor  # describe them (VLM backbone by default)
```

## Setup
`uv sync`, then put `GROQ_API_KEY` (and optionally `GROQ_API_KEY2/3` as fallbacks) in
`.env`. The ESA-Mission1 dataset and the ChatTS checkpoint are not part of this repo.

Checks: `uv run pytest && uv run ruff check src tests && uv run mypy src`
