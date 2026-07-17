"""Timeseries labelling — Part 2 of the project.

Given points of interest on a channel, `TimeSeriesDescriptor` (descriptor.py) asks
ChatTS to describe the signal at three nested granularities (coarse → medium → fine),
chaining each answer into the next prompt, and saves (start, end, description, label).
The points come from `ool.py`: rule-based out-of-limits detection on rolling
median / variance / frequency features over the ESA training telemetry
(run with `python -m src.labelling.run_ool`, then `python -m src.labelling.run_descriptor`).
"""
