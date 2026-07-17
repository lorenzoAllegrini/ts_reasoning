"""TimeSeriesDescriptor — multi-granularity, MULTIVARIATE descriptions of OOL points.

For each point of interest, the model sees the time-aligned windows of the most active
channels of the primary channel's subsystem together (ESA anomalies are mostly
multivariate), at THREE nested granularities — COARSE (± LABEL_COARSE_HALF), MEDIUM
(± LABEL_MEDIUM_HALF) and FINE (± LABEL_FINE_HALF). The three stages are INDEPENDENT:
no stage receives another stage's text (chaining made the medium stage anchor on and
parrot the coarse description instead of describing the new scale). The three texts are
stored side by side per point; combining them into a knowledge base happens downstream.

Deliberate constraints: prompts NEVER point the model at the segment centre nor at which
channel triggered the point — it must notice salient behaviour on its own. The FINE
prompt still forbids assumptions about anything outside its narrow excerpt.

Two interchangeable backbones:
  · TimeSeriesDescriptor     — ChatTS reads the RAW SERIES (full resolution, no
    subsampling; see config.CHATTS_HARD_MAX_POINTS) via `chat(system, user, series)`.
  · VLMTimeSeriesDescriptor  — a Groq vision LM describes RENDERED PLOTS, at most
    VLM_BATCH_SIZE subplots per call, sequential batches for more channels.

`run()` (shared) executes the stages for every input point and appends records
(start, end, description, label) to a CSV (rewritten per point; resumable). No torch
import in this module; backends are injected (fakes in tests).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class ChatTSBackend(Protocol):
    """The one capability the ChatTS descriptor needs."""

    def chat(
        self, system: str, user: str, series: list[np.ndarray], max_new_tokens: int | None = None
    ) -> str: ...


class VLMBackend(Protocol):
    """The one capability the VLM descriptor needs: (system, user, plot PNG) → text."""

    def describe(self, system: str, user: str, png: bytes) -> str: ...


class SubsystemPoint(TypedDict):
    """One point of interest, time-aligned across the subsystem's channels.

    `centers` maps every channel to ITS OWN index of the same instant (channels are
    resampled independently, so the caller aligns them by timestamp). `primary` is the
    channel the point was detected on — recorded in the output, never told to ChatTS.
    """

    primary: str
    centers: dict[str, int]
    label: int | None


class DescriptionRecord(TypedDict):
    """One labelled region: the primary channel's FINE window + the chained description."""

    channel: str  # the primary (detection) channel
    channels: str  # every channel ChatTS saw, "+"-joined
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

COARSE_USER = """\
I have {k} time-aligned telemetry channels ({channels}), each {n} points: {placeholders}
Describe what each channel looks like over this stretch, and which channels behave
similarly."""

MEDIUM_USER = """\
I have {k} time-aligned telemetry channels ({channels}) over a short stretch
({n} points each): {placeholders}
Describe what each channel looks like here, including the finer detail visible at this
scale (exact shape of features, sharp transitions, small oscillations), and which
channels behave similarly."""

FINE_USER = """\
I have a very short excerpt of {k} time-aligned telemetry channels ({channels}),
{n} points each: {placeholders}
Describe exactly what happens inside this excerpt for each channel — the shape of any
feature, its amplitude relative to the local level, and how it begins and ends — and
which channels change at the same time. This is only a small piece of the signal, so
do not make assumptions about what lies outside it; describe just what is inside it."""


class TimeSeriesDescriptor:
    """Runs the coarse → medium → fine multivariate description chain per point."""

    def __init__(
        self,
        backend: ChatTSBackend,
        coarse_half: int = config.LABEL_COARSE_HALF,
        medium_half: int = config.LABEL_MEDIUM_HALF,
        fine_half: int = config.LABEL_FINE_HALF,
        max_new_tokens: int = config.LABEL_MAX_NEW_TOKENS,
    ) -> None:
        if not coarse_half > medium_half > fine_half > 0:
            raise ValueError("windows must nest: coarse_half > medium_half > fine_half > 0")
        self.backend = backend
        self.coarse_half = coarse_half
        self.medium_half = medium_half
        self.fine_half = fine_half
        self.max_new_tokens = max_new_tokens

    # ── windows ───────────────────────────────────────────────────────────────
    @staticmethod
    def _window(x: np.ndarray, center: int, half: int) -> tuple[np.ndarray, int, int]:
        start = max(0, center - half)
        end = min(x.size, center + half)
        return x[start:end], start, end

    def _stage_series(
        self, series: dict[str, np.ndarray], centers: dict[str, int], half: int
    ) -> tuple[list[str], list[np.ndarray]]:
        """Per-channel windows for one granularity, in deterministic (sorted) channel
        order — the primary is NOT first, so its position gives ChatTS no hint."""
        names = sorted(centers)
        return names, [self._window(series[ch], centers[ch], half)[0] for ch in names]

    # ── one point ─────────────────────────────────────────────────────────────
    _TEMPLATES = {"coarse": COARSE_USER, "medium": MEDIUM_USER, "fine": FINE_USER}

    def _stage_user(self, stage: str, names: list[str], n_points: int, placeholders: str) -> str:
        return self._TEMPLATES[stage].format(
            k=len(names), channels=", ".join(names), n=n_points, placeholders=placeholders
        )

    def _describe_stage(self, stage: str, names: list[str], segs: list[np.ndarray]) -> str:
        """One granularity level via ChatTS: raw series + <ts><ts/> placeholders."""
        holders = " ".join("<ts><ts/>" for _ in names)
        user = self._stage_user(stage, names, segs[0].size, holders)
        return self.backend.chat(DESCRIBER_SYSTEM, user, segs, self.max_new_tokens)

    def describe_point(
        self, series: dict[str, np.ndarray], centers: dict[str, int]
    ) -> dict[str, str]:
        """Three INDEPENDENT multivariate descriptions for one point (no chaining —
        each stage sees only its own windows, never another stage's text)."""
        names, coarse_segs = self._stage_series(series, centers, self.coarse_half)
        _, medium_segs = self._stage_series(series, centers, self.medium_half)
        _, fine_segs = self._stage_series(series, centers, self.fine_half)
        return {
            "coarse": self._describe_stage("coarse", names, coarse_segs),
            "medium": self._describe_stage("medium", names, medium_segs),
            "fine": self._describe_stage("fine", names, fine_segs),
        }

    # ── all points → file ─────────────────────────────────────────────────────
    def run(
        self,
        series: dict[str, np.ndarray],
        points: list[SubsystemPoint],
        out_path: str,
        images_dir: str | None = None,
    ) -> pd.DataFrame:
        """Describe every point and persist (start, end, description, label).

        `series` holds the FULL series of every channel involved; each point carries
        its own time-aligned per-channel centers. `start`/`end` are the PRIMARY
        channel's FINE window bounds. The CSV is rewritten after each point
        (crash-safe) and already-described points are skipped, so a long job resumes.
        With `images_dir`, one PNG per granularity (coarse/medium/fine — the exact
        windows ChatTS sees) is saved per point; images are backfilled for
        already-described points too (they are deterministic — no ChatTS call).
        """
        rows = self._existing(out_path)
        done = {(r["channel"], r["center"]) for r in rows}

        for point in points:
            primary, centers = point["primary"], point["centers"]
            center = int(centers[primary])
            if images_dir is not None:
                from src.labelling.plots import (
                    save_point_images,  # noqa: PLC0415 — matplotlib only when asked
                )

                save_point_images(
                    series, centers, primary,
                    {"coarse": self.coarse_half, "medium": self.medium_half, "fine": self.fine_half},
                    images_dir,
                )
            if (primary, center) in done:
                logger.info("describe [%s @ %d]: already in %s — skipped", primary, center, out_path)
                continue
            stages = self.describe_point(series, centers)
            _, start, end = self._window(series[primary], center, self.fine_half)
            description = (
                f"[coarse ±{self.coarse_half}] {stages['coarse']}\n"
                f"[medium ±{self.medium_half}] {stages['medium']}\n"
                f"[fine ±{self.fine_half}] {stages['fine']}"
            )
            rows.append(
                DescriptionRecord(
                    channel=primary, channels="+".join(sorted(centers)), start=start, end=end,
                    center=center, description=description, label=point["label"],
                )
            )
            frame = pd.DataFrame(rows)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out_path, index=False)
            logger.info("describe [%s @ %d]: saved (%d records)", primary, center, len(rows))
        return pd.DataFrame(rows)

    @staticmethod
    def _existing(out_path: str) -> list[DescriptionRecord]:
        if not Path(out_path).exists():
            return []
        frame = pd.read_csv(out_path)
        records: list[DescriptionRecord] = []
        for _, r in frame.iterrows():
            records.append(
                DescriptionRecord(
                    channel=str(r["channel"]), channels=str(r.get("channels", r["channel"])),
                    start=int(r["start"]), end=int(r["end"]), center=int(r["center"]),
                    description=str(r["description"]),
                    label=(None if pd.isna(r["label"]) else int(r["label"])),
                )
            )
        return records


class VLMTimeSeriesDescriptor(TimeSeriesDescriptor):
    """Same three-stage flow (independent stages), backbone = VISION LM on PLOTS.

    Each stage renders the channel windows as line plots and asks the VLM to describe
    them. At most `batch_size` (default 4) subplots go into one call — when a stage has
    more channels, they are described in SEQUENTIAL batches and the batch texts are
    joined as the stage text. Everything else — windows, prompts, CSV format, resume,
    images — is inherited unchanged from TimeSeriesDescriptor.
    """

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
        self.coarse_half = coarse_half
        self.medium_half = medium_half
        self.fine_half = fine_half
        self.max_new_tokens = 0  # unused: the VLM backend caps its own output

    def _describe_stage(self, stage: str, names: list[str], segs: list[np.ndarray]) -> str:
        """One granularity level via the VLM: render ≤batch_size panels per call."""
        from src.labelling.plots import render_batch_png  # noqa: PLC0415 — matplotlib on demand

        parts: list[str] = []
        for i in range(0, len(names), self.batch_size):
            batch_names = names[i : i + self.batch_size]
            batch_segs = segs[i : i + self.batch_size]
            user = self._stage_user(
                stage, batch_names, batch_segs[0].size,
                "shown in the attached plot (one panel per channel)",
            )
            png = render_batch_png(list(zip(batch_names, batch_segs, strict=True)))
            text = self.vlm.describe(DESCRIBER_SYSTEM, user, png)
            parts.append(text if len(names) <= self.batch_size else f"[{', '.join(batch_names)}] {text}")
        return "\n".join(parts)
