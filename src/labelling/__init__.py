"""Part 2 — timeseries labelling.

Rule-based OOL detection over the ESA training telemetry (`ool.py`, run with
`python -m src.labelling.run_ool`), then a Groq vision LM describes each detected
point at three independent granularities from rendered plots (`descriptor.py` +
`vlm.py` + `plots.py`, run with `python -m src.labelling.run_descriptor`).
Sibling part: `src.agentic` (the interpretation loop).
"""
