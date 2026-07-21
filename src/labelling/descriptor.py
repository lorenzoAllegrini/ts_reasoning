"""TimeSeriesDescriptor — a vision LM describes each OOL point at three granularities.

For each point of interest, the most active channels of its subsystem are rendered as
line plots (time-aligned windows) and described by a Groq vision LM at three nested,
INDEPENDENT granularities — coarse (± LABEL_COARSE_HALF), medium (± LABEL_MEDIUM_HALF)
and fine (± LABEL_FINE_HALF). No stage sees another stage's text (chaining made stages
parrot each other); the three texts are stored side by side per point. At most
VLM_BATCH_SIZE subplots go into one call — more channels are described in sequential
batches.

Deliberate constraints: neither the prompts nor the rendered plots point the model at
the segment centre or at the detection channel — it must notice salient behaviour on
its own. The FINE prompt forbids assumptions about anything outside its excerpt.

`run()` describes every input point and appends records (start, end, description,
label) to a CSV — rewritten after each point and resumable (already-described points
are skipped; their diagnostic images are still backfilled on demand).

The backend is anything with `describe(system, user, png) -> str` — `vlm.GroqVLM` in
production, a fake in tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class VLMBackend(Protocol):
    """The one capability the descriptor needs: (system, user, plot PNG) → text."""

    def describe(self, system: str, user: str, png: bytes) -> str: ...


class SubsystemPoint(TypedDict):
    """One point of interest, time-aligned across the subsystem's channels.

    `centers` maps every channel to ITS OWN index of the same instant (channels are
    resampled independently, so the caller aligns them by timestamp). `primary` is the
    channel the point was detected on — recorded in the output, never told to the model.
    """

    primary: str
    centers: dict[str, int]
    label: int | None


class DescriptionRecord(TypedDict):
    """One labelled region: the primary channel's FINE window + the three descriptions."""

    channel: str  # the primary (detection) channel
    channels: str  # every channel the model saw, "+"-joined
    start: int
    end: int
    center: int
    description: str
    label: int | None


DESCRIBER_SYSTEM = """\
You describe multi-channel spacecraft telemetry. For each channel say what it looks like:
its level and trend, whether it is periodic or modulated, how variable it is, and any
clear local feature (step, spike, ramp, plateau, dropout, oscillation) with its rough
position. Then say which channels behave alike and which stand out. Describe only what is
visible; do not judge whether anything is an anomaly."""

STAGE_PROMPTS = {
    "coarse": """\
I have {k} time-aligned telemetry channels ({channels}), each {n} points, shown in the
attached plot (one panel per channel).
Describe what each channel looks like over this stretch, and which channels behave
similarly.""",
    "medium": """\
I have {k} time-aligned telemetry channels ({channels}) over a short stretch
({n} points each), shown in the attached plot (one panel per channel).
Describe what each channel looks like here, including the finer detail visible at this
scale (exact shape of features, sharp transitions, small oscillations), and which
channels behave similarly.""",
    "fine": """\
I have a very short excerpt of {k} time-aligned telemetry channels ({channels}),
{n} points each, shown in the attached plot (one panel per channel).
Describe exactly what happens inside this excerpt for each channel — the shape of any
feature, its amplitude relative to the local level, and how it begins and ends — and
which channels change at the same time. This is only a small piece of the signal, so
do not make assumptions about what lies outside it; describe just what is inside it.""",
}


class TimeSeriesDescriptor:
    """Renders per-stage plots and asks the VLM for three independent descriptions."""

    def __init__(
        self,
        vlm: VLMBackend,
        batch_size: int = config.VLM_BATCH_SIZE,
        coarse_half: int = config.LABEL_COARSE_HALF,
        medium_half: int = config.LABEL_MEDIUM_HALF,
        fine_half: int = config.LABEL_FINE_HALF,
    ) -> None:
        if not coarse_half > medium_half > fine_half > 0:
            raise ValueError("windows must nest: coarse_half > medium_half > fine_half > 0")
        if batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")
        self.vlm = vlm
        self.batch_size = batch_size
        self.halves = {"coarse": coarse_half, "medium": medium_half, "fine": fine_half}

    # ── windows ───────────────────────────────────────────────────────────────
    @staticmethod
    def _window(x: np.ndarray, center: int, half: int) -> tuple[np.ndarray, int, int]:
        start = max(0, center - half)
        end = min(x.size, center + half)
        return x[start:end], start, end

    # ── one stage: render ≤batch_size panels per call, join batch texts ───────
    def _describe_stage(
        self, stage: str, series: dict[str, np.ndarray], centers: dict[str, int]
    ) -> str:
        from src.labelling.plots import render_batch_png  # noqa: PLC0415 — matplotlib on demand

        names = sorted(centers)  # deterministic order; the primary is never first on purpose
        segs = [self._window(series[ch], centers[ch], self.halves[stage])[0] for ch in names]
        parts: list[str] = []
        for i in range(0, len(names), self.batch_size):
            batch_names, batch_segs = names[i : i + self.batch_size], segs[i : i + self.batch_size]
            user = STAGE_PROMPTS[stage].format(
                k=len(batch_names), channels=", ".join(batch_names), n=batch_segs[0].size
            )
            png = render_batch_png(list(zip(batch_names, batch_segs, strict=True)))
            text = self.vlm.describe(DESCRIBER_SYSTEM, user, png)
            parts.append(text if len(names) <= self.batch_size else f"[{', '.join(batch_names)}] {text}")
        return "\n".join(parts)

    def describe_point(
        self, series: dict[str, np.ndarray], centers: dict[str, int]
    ) -> dict[str, str]:
        """Three INDEPENDENT multivariate descriptions for one point."""
        return {stage: self._describe_stage(stage, series, centers) for stage in self.halves}

    # ── all points → file ─────────────────────────────────────────────────────
    def run(
        self,
        series: dict[str, np.ndarray],
        points: list[SubsystemPoint],
        out_path: str,
        images_dir: str | None = None,
    ) -> pd.DataFrame:
        """Describe every point and persist (start, end, description, label).

        `start`/`end` are the PRIMARY channel's FINE window bounds. The CSV is
        rewritten after each point (crash-safe) and already-described points are
        skipped, so a long job resumes. With `images_dir`, one diagnostic PNG per
        granularity is saved per point (backfilled for skipped points too — they
        are deterministic, no model call)."""
        rows = self._existing(out_path)
        done = {(r["channel"], r["center"]) for r in rows}

        for point in points:
            primary, centers = point["primary"], point["centers"]
            center = int(centers[primary])
            if images_dir is not None:
                from src.labelling.plots import save_point_images  # noqa: PLC0415

                save_point_images(series, centers, primary, self.halves, images_dir)
            if (primary, center) in done:
                logger.info("describe [%s @ %d]: already in %s — skipped", primary, center, out_path)
                continue

            stages = self.describe_point(series, centers)
            _, start, end = self._window(series[primary], center, self.halves["fine"])
            description = "\n".join(
                f"[{stage} ±{half}] {stages[stage]}" for stage, half in self.halves.items()
            )
            rows.append(
                DescriptionRecord(
                    channel=primary, channels="+".join(sorted(centers)), start=start, end=end,
                    center=center, description=description, label=point["label"],
                )
            )
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            logger.info("describe [%s @ %d]: saved (%d records)", primary, center, len(rows))
        return pd.DataFrame(rows)

    @staticmethod
    def _existing(out_path: str) -> list[DescriptionRecord]:
        if not Path(out_path).exists():
            return []
        frame = pd.read_csv(out_path)
        return [
            DescriptionRecord(
                channel=str(r["channel"]), channels=str(r["channels"]),
                start=int(r["start"]), end=int(r["end"]), center=int(r["center"]),
                description=str(r["description"]),
                label=(None if pd.isna(r["label"]) else int(r["label"])),
            )
            for _, r in frame.iterrows()
        ]
