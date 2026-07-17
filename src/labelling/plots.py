"""Per-point diagnostic images for the labelling pipeline — pure matplotlib, no LLM.

For every OOL point, three PNGs are saved (coarse / medium / fine), each showing the
SAME time-aligned windows the descriptor feeds to ChatTS: one small-multiple panel per
subsystem channel, x = offset in samples from the point. These images are for HUMAN
review of the produced labels — they are never shown to ChatTS — so, unlike the
prompts, they MAY mark the point (dashed line) and the detection channel (highlighted
title).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless — never require a display
import matplotlib.pyplot as plt  # noqa: E402 — must follow the backend selection

logger = logging.getLogger(__name__)

_MAX_COLS = 5
PRIMARY_COLOR = "#c62828"


def render_batch_png(named_segs: list[tuple[str, np.ndarray]]) -> bytes:
    """One PNG with ≤4 vertically-stacked panels — the image a VLM describes.

    Unlike the human-review images, this rendering is NEUTRAL: no centre line, no
    highlighted channel (the model must notice salient behaviour on its own, same
    constraint as the prompts). X axis = local sample index of the window."""
    import io  # noqa: PLC0415

    if not 1 <= len(named_segs) <= 4:
        raise ValueError(f"a VLM batch renders 1–4 panels, got {len(named_segs)}")
    fig, axes = plt.subplots(len(named_segs), 1, figsize=(8.0, 2.1 * len(named_segs)), squeeze=False)
    for ax, (name, seg) in zip(axes.flat, named_segs, strict=True):
        ax.plot(np.arange(seg.size), seg, linewidth=0.9)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def save_point_images(
    series: dict[str, np.ndarray],
    centers: dict[str, int],
    primary: str,
    halves: dict[str, int],
    out_dir: str,
    annotate: bool = True,
) -> list[str]:
    """Write one PNG per granularity for this point; returns the created paths.

    `halves` maps stage name → window half-width (e.g. {"coarse": 512, ...}); the
    windows drawn are exactly the ones the descriptor cuts (clamped at the edges).
    Files: {out_dir}/{primary}_{center}_{stage}.png — overwritten if present.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    names = sorted(centers)
    center = int(centers[primary])
    paths: list[str] = []

    for stage, half in halves.items():
        ncols = min(_MAX_COLS, len(names))
        nrows = math.ceil(len(names) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 1.9 * nrows), squeeze=False, sharex=True
        )
        for ax, name in zip(axes.flat, names, strict=False):
            x, c = series[name], int(centers[name])
            start, end = max(0, c - half), min(x.size, c + half)
            ax.plot(np.arange(start, end) - c, x[start:end], linewidth=0.7)
            is_primary = name == primary
            ax.set_title(
                name, fontsize=8,
                color=PRIMARY_COLOR if (annotate and is_primary) else "black",
                fontweight="bold" if (annotate and is_primary) else "normal",
            )
            if annotate:
                ax.axvline(0, color=PRIMARY_COLOR, linestyle="--", linewidth=0.6, alpha=0.6)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(names) :]:  # hide unused grid cells
            ax.set_visible(False)
        fig.suptitle(f"{primary} @ {center} — {stage} ±{half} samples", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        path = str(Path(out_dir) / f"{primary}_{center}_{stage}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    logger.info("images [%s @ %d]: %d saved in %s", primary, center, len(paths), out_dir)
    return paths
