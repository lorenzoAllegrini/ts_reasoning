"""Rule-based OOL (out-of-limits) point detection — pure numpy, no LLM, no torch.

Each channel is scanned with strided rolling windows; every window yields THREE
features — 1. median (level), 2. std (variance), 3. zero-crossing rate of the
median-detrended window (frequency) — each turned into a robust z-score against the
channel's own global distribution of that feature. A window is a CANDIDATE when its
strongest |z| clears a hard floor (config.OOL_Z_FLOOR).

Selection (`select_balanced`) is PER CHANNEL and balanced PER SUBSYSTEM: z values are
never compared across channels (their feature distributions have different tail
weights — a global top-k collapses onto a couple of heavy-tailed channels). Every
subsystem gets an equal quota of the budget (total strictly < max_total) and fills it
round-robin over its channels, each contributing its next-best candidate by its OWN z;
the weakest accepted z of a channel is that channel's effective OOL threshold. The
"distinct points" guarantee stays global: every accepted pair is ≥ `min_separation_s`
apart in time, on any channel — simultaneous hits are the same (multivariate) anomaly.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from src import config


class OOLCandidate(TypedDict):
    """One strided window whose strongest feature cleared the z floor."""

    channel: str
    center: int  # sample index of the window centre (channel-local)
    time_s: float  # epoch seconds of the centre (comparable ACROSS channels)
    feature: str  # which feature fired: "median" | "std" | "zcr"
    z: float  # the robust |z| of that feature
    label: int  # ground truth: window overlaps a labelled anomaly (evaluation only)


def window_features(
    x: np.ndarray, window: int = config.OOL_WINDOW, hop: int = config.OOL_HOP
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(window centres, {feature name → per-window value}) for strided windows."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < window:
        return np.empty(0, dtype=int), {"median": np.empty(0), "std": np.empty(0), "zcr": np.empty(0)}
    views = np.lib.stride_tricks.sliding_window_view(x, window)[::hop]
    centers = np.arange(views.shape[0]) * hop + window // 2

    median = np.median(views, axis=1)
    std = views.std(axis=1)
    # frequency: sign changes of the median-detrended window, per sample
    detrended_sign = np.sign(views - median[:, None])
    zcr = (np.diff(detrended_sign, axis=1) != 0).sum(axis=1) / (window - 1)
    return centers, {"median": median, "std": std, "zcr": zcr}


def robust_z(values: np.ndarray) -> np.ndarray:
    """|z| against the series' own median/MAD (1.4826·MAD ≈ σ). A constant feature
    (MAD = 0 and no spread at all) yields all-zero z — nothing can be out of limits."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = values.std() or np.inf  # fallback for quantised features; inf → z 0
    return np.abs(values - center) / scale


def channel_candidates(
    x: np.ndarray,
    channel: str,
    time_s: np.ndarray,
    anomaly_intervals: list[tuple[int, int]],
    window: int = config.OOL_WINDOW,
    hop: int = config.OOL_HOP,
    z_floor: float = config.OOL_Z_FLOOR,
    min_std: float = config.OOL_MIN_STD,
) -> list[OOLCandidate]:
    """All windows of one channel whose strongest feature-z clears the floor AND that
    carry real signal amplitude (window std ≥ `min_std`).

    The amplitude gate is essential: on near-constant channels the robust-z of a feature
    blows up on negligible absolute variation (subsystem_1 fired z≈11 on std≈0.003), so a
    RELATIVE threshold alone scrapes quantisation noise. `time_s` is the per-sample
    epoch-seconds array (channels differ in length / gap structure, so cross-channel
    separation is enforced on TIME, not index). `anomaly_intervals` is ground truth used
    ONLY to attach an evaluation label.
    """
    centers, feats = window_features(x, window, hop)
    if centers.size == 0:
        return []
    names = list(feats)
    z_matrix = np.stack([robust_z(feats[name]) for name in names])  # (3, n_windows)
    best_feature = z_matrix.argmax(axis=0)
    best_z = z_matrix.max(axis=0)

    half = window // 2
    has_amplitude = feats["std"] >= min_std  # the window contains real variation, not flat-channel noise
    keep = np.flatnonzero((best_z >= z_floor) & has_amplitude)
    out: list[OOLCandidate] = []
    for i in keep:
        c = int(centers[i])
        lo, hi = c - half, c + half
        label = int(any(s <= hi and e >= lo for s, e in anomaly_intervals))
        out.append(
            OOLCandidate(
                channel=channel, center=c, time_s=float(time_s[min(c, time_s.size - 1)]),
                feature=names[int(best_feature[i])], z=float(best_z[i]), label=label,
            )
        )
    return out


def select_balanced(
    candidates: list[OOLCandidate],
    subsystem_by_channel: dict[str, str],
    max_total: int = config.OOL_MAX_POINTS,
    min_separation_s: float = config.OOL_MIN_SEPARATION_S,
) -> tuple[list[OOLCandidate], dict[str, float], dict[str, int]]:
    """Per-channel OOL selection, balanced so every SUBSYSTEM gets ~the same number
    of points and the grand total stays STRICTLY below `max_total`.

    Raw z values are NEVER compared across channels — feature distributions have
    different tail weights per channel, so a global top-k collapses onto the few
    channels with the heaviest tails. Instead each subsystem receives an equal quota
    ((max_total-1) // n_subsystems) and fills it by ROUND-ROBIN over its channels:
    each channel, in turn, contributes its next-best candidate BY ITS OWN z. The z of
    the weakest accepted candidate of a channel is that channel's effective OOL
    threshold — i.e. the parameters are per-channel, as they must be.

    The time-separation guarantee stays GLOBAL (≥ min_separation_s from every other
    accepted point, any channel or subsystem): simultaneous hits are the same anomaly.

    Returns (selected, per-channel effective thresholds, per-subsystem counts).
    """
    if not candidates:
        return [], {}, {}

    # subsystem → channel → candidates sorted by that channel's OWN z, best first
    queues: dict[str, dict[str, list[OOLCandidate]]] = {}
    for cand in candidates:
        sub = subsystem_by_channel.get(cand["channel"], "unknown")
        queues.setdefault(sub, {}).setdefault(cand["channel"], []).append(cand)
    for channels in queues.values():
        for queue in channels.values():
            queue.sort(key=lambda cand: cand["z"])  # ascending → pop() yields the best

    quota = max(1, (max_total - 1) // len(queues))  # equal per subsystem, total < max_total
    selected: list[OOLCandidate] = []
    times: list[float] = []

    def admissible(cand: OOLCandidate) -> bool:
        return all(abs(cand["time_s"] - t) >= min_separation_s for t in times)

    for sub in sorted(queues):
        channels = queues[sub]
        taken = 0
        while taken < quota and any(channels.values()):
            for channel in sorted(channels):
                if taken >= quota:
                    break
                queue = channels[channel]
                while queue:  # this channel's turn: its best still-admissible candidate
                    cand = queue.pop()
                    if admissible(cand):
                        selected.append(cand)
                        times.append(cand["time_s"])
                        taken += 1
                        break  # inadmissible ones are the same anomaly as a chosen point → dropped

    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cand in selected:
        thresholds[cand["channel"]] = min(thresholds.get(cand["channel"], float("inf")), cand["z"])
        sub = subsystem_by_channel.get(cand["channel"], "unknown")
        counts[sub] = counts.get(sub, 0) + 1
    return selected, thresholds, counts
