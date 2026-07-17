"""Data access — resolve the AgentState `data` identifier to channel arrays, and
extract windows from the real ESA-ADB benchmark.

The state carries `data: str` (a path to an .npz of channel arrays); nodes call
here to obtain numpy arrays, keeping the statistical tools pure (they receive
arrays, never a path). Ground-truth labels are stored under a reserved
"__labels__" key and are reachable ONLY via get_labels(), which no node calls —
labels evaluate the system, they never inform it (§0.5, anti-cheating).

The ESA adapter reads the same source files as the torch-coupled `spaceai` loader
(pickled per-channel DataFrames + labels.csv), resamples to a uniform grid, and
writes the same .npz format — a lightweight, torch-free path for pulling a window.
"""

from __future__ import annotations

import functools
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_RESERVED_PREFIX = "__"
_LABELS_KEY = "__labels__"

# ── ESA-ADB Mission-1 ──────────────────────────────────────────────────────
ESA_MISSION1_DIR = Path("space-ai/datasets/ESA-Mission1/ESA-Mission1")
ESA_DEFAULT_RESAMPLE = "2min"  # uniform grid; the tools assume uniform sampling
ESA_CONTEXT_BEFORE = pd.Timedelta("1D")  # nominal context kept before the anomaly
ESA_CONTEXT_AFTER = pd.Timedelta("6h")  # nominal context kept after the anomaly


@functools.cache
def _load(path: str) -> dict[str, np.ndarray]:
    """Load and cache all arrays in the .npz (write-once, read-many)."""
    with np.load(path) as npz:
        return {key: np.asarray(npz[key], dtype=float) for key in npz.files}


def list_channels(path: str) -> list[str]:
    """Channel names available to the system (reserved keys excluded)."""
    return sorted(k for k in _load(path) if not k.startswith(_RESERVED_PREFIX))


def get_channel(path: str, channel: str) -> np.ndarray:
    """Return the 1-D array for `channel`."""
    if channel.startswith(_RESERVED_PREFIX):
        raise KeyError(f"{channel!r} is reserved and not a channel")
    arrays = _load(path)
    if channel not in arrays:
        raise KeyError(f"unknown channel {channel!r}; available: {list_channels(path)}")
    return arrays[channel]


def series_length(path: str) -> int:
    """Length of the (equal-length) channels."""
    arrays = _load(path)
    lengths = {v.shape[0] for k, v in arrays.items() if not k.startswith(_RESERVED_PREFIX)}
    return int(next(iter(lengths))) if lengths else 0


def get_labels(path: str) -> np.ndarray | None:
    """Ground-truth anomaly labels — for EVALUATION ONLY. Never call from a node."""
    return _load(path).get(_LABELS_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# ESA-ADB adapter (was esa.py) — torch-free window extraction into the .npz format.
# ⚠️ Anti-leakage: labels.csv anomaly intervals are used ONLY to place the window
# and to write the reserved __labels__ array (evaluation); they never inform the loop.
# ─────────────────────────────────────────────────────────────────────────────
def _load_esa_channel_frame(channel_id: str, mission_dir: Path) -> pd.DataFrame:
    """Read one channel's pickled DataFrame (DatetimeIndex + single value column)."""
    zip_path = mission_dir / "channels" / f"{channel_id}.zip"
    with zipfile.ZipFile(zip_path) as z, z.open(channel_id) as f:
        frame: pd.DataFrame = pd.read_pickle(f)
    return frame


def esa_anomaly_intervals(channel_id: str, mission_dir: Path = ESA_MISSION1_DIR) -> pd.DataFrame:
    labels = pd.read_csv(mission_dir / "labels.csv")
    out: pd.DataFrame = labels[labels["Channel"] == channel_id].copy()
    out["StartTime"] = pd.to_datetime(out["StartTime"]).dt.tz_localize(None)
    out["EndTime"] = pd.to_datetime(out["EndTime"]).dt.tz_localize(None)
    return out.sort_values("StartTime").reset_index(drop=True)


def build_esa_window_npz(
    out_path: str,
    channel_id: str = "channel_12",
    anomaly_rank: int = 0,
    context_before: pd.Timedelta = ESA_CONTEXT_BEFORE,
    context_after: pd.Timedelta = ESA_CONTEXT_AFTER,
    resample_rule: str = ESA_DEFAULT_RESAMPLE,
    mission_dir: Path = ESA_MISSION1_DIR,
) -> str:
    """Extract a window around the `anomaly_rank`-th labelled anomaly of `channel_id`
    (with nominal context before/after), resample to a uniform grid, and save an
    .npz with the channel values and a `__labels__` array. Returns out_path."""
    intervals = esa_anomaly_intervals(channel_id, mission_dir)
    if anomaly_rank >= len(intervals):
        raise IndexError(f"{channel_id} has {len(intervals)} anomalies; rank {anomaly_rank} invalid")
    anomaly = intervals.iloc[anomaly_rank]
    window_start = anomaly["StartTime"] - context_before
    window_end = anomaly["EndTime"] + context_after

    frame = _load_esa_channel_frame(channel_id, mission_dir)
    window = frame.loc[window_start:window_end]  # fast DatetimeIndex slice
    series = window[channel_id].resample(resample_rule).mean().ffill().bfill()

    values = series.to_numpy(dtype=float)
    timestamps = series.index
    labels = np.zeros(len(values), dtype=float)
    for _, row in intervals.iterrows():
        labels[(timestamps >= row["StartTime"]) & (timestamps <= row["EndTime"])] = 1.0

    np.savez(out_path, **{channel_id: values, "__labels__": labels})  # type: ignore[arg-type]
    return out_path


def build_esa_window(
    out_path: str,
    channels: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    resample_rule: str = ESA_DEFAULT_RESAMPLE,
    mission_dir: Path = ESA_MISSION1_DIR,
) -> str:
    """Multi-channel ESA window on a common uniform grid, saved as an .npz.

    All channels are reindexed to one grid so they are equal-length (the tools and
    ChatTS need aligned series). `__labels__` marks points inside any labelled
    anomaly interval for ANY of the channels — evaluation only (§0.5)."""
    grid = pd.date_range(window_start, window_end, freq=resample_rule)
    arrays: dict[str, np.ndarray] = {}
    for ch in channels:
        frame = _load_esa_channel_frame(ch, mission_dir)
        series = frame.loc[window_start:window_end][ch].resample(resample_rule).mean()
        arrays[ch] = series.reindex(grid).ffill().bfill().to_numpy(dtype=float)

    labels = np.zeros(len(grid), dtype=float)
    all_labels = pd.read_csv(mission_dir / "labels.csv")  # evaluation only
    for ch in channels:
        for _, row in all_labels[all_labels["Channel"] == ch].iterrows():
            st = pd.to_datetime(row["StartTime"]).tz_localize(None)
            et = pd.to_datetime(row["EndTime"]).tz_localize(None)
            labels[(grid >= st) & (grid <= et)] = 1.0

    np.savez(out_path, __labels__=labels, **arrays)  # type: ignore[arg-type]
    return out_path
