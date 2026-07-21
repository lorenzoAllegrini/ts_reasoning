"""Describe the OOL points — a Groq vision LM sees the detection channel's subsystem.

    uv run python -m src.labelling.run_descriptor                 # all OOL points
    uv run python -m src.labelling.run_descriptor --limit 2       # smoke run
    uv run python -m src.labelling.run_descriptor --channel channel_41

Reads labelling_out/ool_points.csv (from run_ool). For every point, a same-subsystem
pool of channels (channels.csv; Group first, then numeric proximity, ≤ LABEL_POOL_SIZE)
is loaded with the spaceai preprocessing and time-aligned on the point's timestamp —
channels are resampled independently, so alignment is by time, not index. Only the
primary + the most ACTIVE siblings (largest window std, ≤ LABEL_MAX_SERIES) are shown
to the model. Loaded channels are cached across points; the output CSV is rewritten
after every point and already-described points are skipped, so the job is resumable.
Ground-truth labels ride along for evaluation only — never shown to the model.

Output: labelling_out/descriptions.csv (channel, channels, start, end, center, description, label)
        labelling_out/images/{channel}_{center}_{coarse|medium|fine}.png (human review)
"""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src import config
from src.labelling.descriptor import SubsystemPoint, TimeSeriesDescriptor
from src.labelling.run_ool import OUT_DIR
from src.labelling.vlm import GroqVLM

logger = logging.getLogger(__name__)


def _channel_number(ch: str) -> int:
    return int(ch.rsplit("_", 1)[1])


def select_subsystem_channels(
    primary: str, table: pd.DataFrame, pool_size: int = config.LABEL_POOL_SIZE
) -> list[str]:
    """The same-subsystem POOL to load for `primary`, capped at `pool_size`.

    Deterministic priority when the subsystem exceeds `pool_size`: the primary, then its
    Group members, then the rest of the subsystem by numeric channel proximity (nearby
    channel ids are physically adjacent parameters). Always within one subsystem. The
    subset actually shown to the model is chosen later by `most_active_channels`. Pure."""
    row = table.loc[table["Channel"] == primary]
    if row.empty:
        return [primary]
    subsystem, group = row.iloc[0]["Subsystem"], row.iloc[0]["Group"]
    members = table.loc[table["Subsystem"] == subsystem]

    def priority(r: pd.Series) -> tuple[int, int]:
        if r["Channel"] == primary:
            return (0, 0)
        return (1 if r["Group"] == group else 2, abs(_channel_number(r["Channel"]) - _channel_number(primary)))

    ranked = sorted((r for _, r in members.iterrows()), key=priority)
    return sorted((r["Channel"] for r in ranked[:pool_size]), key=_channel_number)


def most_active_channels(
    series: dict[str, np.ndarray],
    centers: dict[str, int],
    primary: str,
    k: int = config.LABEL_MAX_SERIES,
    half: int = config.LABEL_COARSE_HALF,
) -> list[str]:
    """The primary + the (k-1) siblings with the largest std in the coarse window.

    Vision models degrade on many panels, and near-flat siblings add noise, not
    signal — only the channels that actually MOVE around the point are shown. Pure."""
    def activity(ch: str) -> float:
        c = centers[ch]
        w = series[ch][max(0, c - half) : c + half]
        return float(w.std()) if w.size else 0.0

    others = sorted((ch for ch in series if ch != primary), key=activity, reverse=True)
    return sorted([primary, *others[: max(0, k - 1)]], key=_channel_number)


class ChannelCache:
    """Loads each ESA training channel once (float32 data + epoch-second timestamps).

    spaceai's per-channel preprocessing expands a ~7M-row pickle — big transients; the
    ESA object is dropped and a GC forced right after keeping the two arrays, so
    transients never accumulate across a subsystem's channels."""

    def __init__(self, root: str, max_cached: int = 64) -> None:
        from spaceai.data.esa import ESAMissions  # noqa: PLC0415 — torch-heavy, script-only

        self.root = root
        self.mission = ESAMissions.MISSION_1.value
        self.max_cached = max_cached
        self._store: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        if channel not in self._store:
            from spaceai.data.esa import ESA  # noqa: PLC0415

            esa = ESA(root=self.root, mission=self.mission, channel_id=channel, train=True, download=False)
            data = np.ascontiguousarray(esa.data[:, 0], dtype=np.float32)
            times = np.asarray(esa.timestamps).astype("datetime64[s]").astype(np.int64)
            if len(self._store) >= self.max_cached:
                self._store.pop(next(iter(self._store)))  # evict oldest
            self._store[channel] = (data, times)
            del esa
            gc.collect()
            logger.info("loaded %s (%d samples)", channel, times.size)
        return self._store[channel]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Describe OOL points with a vision LM at three granularities")
    parser.add_argument("--points", default=str(OUT_DIR / "ool_points.csv"), help="CSV from run_ool")
    parser.add_argument("--out", default=str(OUT_DIR / "descriptions.csv"))
    parser.add_argument("--root", default="space-ai/datasets", help="spaceai dataset root")
    parser.add_argument("--channel", default=None, help="restrict to one primary channel")
    parser.add_argument("--limit", type=int, default=None, help="describe at most N points (smoke runs)")
    parser.add_argument("--images-dir", default=str(OUT_DIR / "images"), help="where the per-point PNGs go")
    parser.add_argument("--no-images", action="store_true", help="skip saving the PNGs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("urllib3", "httpx", "groq"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    load_dotenv()  # GROQ_API_KEY(2/3) for the VLM

    points = pd.read_csv(args.points, parse_dates=["timestamp"])
    if args.channel:
        points = points[points["channel"] == args.channel]
    if args.limit is not None:
        points = points.head(args.limit)
    if points.empty:
        logger.info("no OOL points to describe (%s)", args.points)
        return 0

    table = pd.read_csv(config.CHANNELS_CSV)
    cache = ChannelCache(args.root)
    descriptor = TimeSeriesDescriptor(GroqVLM())
    logger.info("backbone: %s (batches of ≤%d subplots)", config.GROQ_VLM_MODEL, config.VLM_BATCH_SIZE)

    for _, row in points.iterrows():
        primary = str(row["channel"])
        instant = float(pd.Timestamp(row["timestamp"]).timestamp())
        pool = select_subsystem_channels(primary, table)

        pool_series: dict[str, np.ndarray] = {}
        pool_centers: dict[str, int] = {}
        for ch in pool:
            x, times = cache.get(ch)
            pool_series[ch] = x
            pool_centers[ch] = int(np.searchsorted(times, instant))  # time-aligned, per-channel index
        active = most_active_channels(pool_series, pool_centers, primary)
        point = SubsystemPoint(
            primary=primary, centers={ch: pool_centers[ch] for ch in active}, label=int(row["label"])
        )
        logger.info("%s @ %s: %d-channel pool → %d most-active: %s",
                    primary, row["timestamp"], len(pool), len(active), active)
        descriptor.run(
            {ch: pool_series[ch] for ch in active}, [point], out_path=args.out,
            images_dir=None if args.no_images else args.images_dir,
        )

    logger.info("descriptions saved → %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
