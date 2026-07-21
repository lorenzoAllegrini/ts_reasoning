# Labelling descriptions — Part 2 experiment outputs

Real descriptions produced by `src/labelling` on ESA-Mission1 telemetry. Each CSV row is
`(channel, [channels,] start, end, center, description, label)`, where `description`
concatenates the three granularity stages `[coarse ±512] … [medium ±128] … [fine ±24]`
(samples around the point). `label` is the ground-truth flag (evaluation only).

## Detection input

- **`ool_points.csv`** — the 196 rule-based OOL points (balanced across subsystems,
  ≥ 1 day apart), the input to every description below.
- **`ool_meta.json`** — per-channel effective z-thresholds, per-subsystem counts,
  parameters, label hit rate (0.38).

## ChatTS backbone (the original attempt — kept for the record)

ChatTS (Qwen3-TS, 16 GB, local MPS) reads the RAW series values.

- **`chatts_univariate_channel_66.csv`** — the **best ChatTS result**: 1–2 channels,
  short output. The fine stage caught the anomaly unprompted ("rapid rise from ~0.38 to
  ~1.14, then falls back"). This is what worked.
- **`chatts_multivariate_channel_61.csv`** — a **degenerate** case: on a point whose 15
  subsystem channels are all near-flat, ChatTS collapses into transcribing raw values
  (`[0.0.0.0.0…]`) — no words.
- **`chatts_multivariate_subsystem3.csv`** — channel_74 / channel_76 with the whole
  subsystem + "be exhaustive" prompts + sampling: the coarse stage degenerates into raw
  numbers / instruction-echoing; the fine stage is usable. This instability across many
  channels + long chained generation is exactly what motivated the backbone switch.

## VLM backbone (the final choice)

A Groq vision LM (qwen3.6-27b) describes rendered PLOTS of the same windows (≤ 4 subplots
per call), three independent stages.

- **`vlm_multivariate_subsystem3.csv`** — the SAME two points
  (channel_74 / channel_76) as `chatts_multivariate_subsystem3.csv`, for a direct
  before/after: structured per-channel + joint prose, correct levels and feature
  positions, and channel_74's labelled anomaly found unprompted ("sharp narrow dropout
  near sample ~600, ~1.0→0.0, square-wave-like dip"). No degeneration.

See `src/labelling/descriptor.py` for the current pipeline (VLM only; ChatTS retired).
