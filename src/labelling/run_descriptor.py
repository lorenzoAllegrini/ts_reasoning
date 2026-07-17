"""Feed the OOL points to the TimeSeriesDescriptor — ChatTS labels each one, seeing
the WHOLE SUBSYSTEM of the detection channel together (multivariate description).

    uv run python -m src.labelling.run_descriptor                 # all OOL points
    uv run python -m src.labelling.run_descriptor --limit 2       # smoke run
    uv run python -m src.labelling.run_descriptor --channel channel_41

Reads labelling_out/ool_points.csv (from run_ool). For every point, ALL channels of
the primary channel's subsystem (channels.csv) are loaded with the same spaceai
preprocessing and their windows time-aligned on the point's timestamp — channels are
resampled independently, so alignment is by time, not index. Subsystems larger than
ChatTS's 30-series input limit are cut deterministically: the primary's Group first,
then numerically-nearest channels. ChatTS (~16 GB) loads once; loaded channels are
cached across points. The output CSV is rewritten after every point and
already-described points are skipped, so the job is resumable. Ground-truth labels
ride along for evaluation only — never shown to ChatTS.

Output: labelling_out/descriptions.csv
        (channel, channels, start, end, center, description, label)
        labelling_out/images/{channel}_{center}_{coarse|medium|fine}.png
        (small-multiple plots of the exact windows ChatTS sees, for human review;
        the detection channel and the point are marked — ChatTS never sees these)
"""

from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src import config
from src.labelling.descriptor import SubsystemPoint, TimeSeriesDescriptor, VLMTimeSeriesDescriptor
from src.labelling.run_ool import OUT_DIR

logger = logging.getLogger(__name__)

CHANNELS_CSV = config.CHANNELS_CSV  # channel → subsystem/group metadata


def _channel_number(ch: str) -> int:
    return int(ch.rsplit("_", 1)[1])


def select_subsystem_channels(
    primary: str, table: pd.DataFrame, pool_size: int = config.LABEL_POOL_SIZE
) -> list[str]:
    """The same-subsystem POOL to load for `primary`, capped at `pool_size`.

    Deterministic priority when the subsystem exceeds `pool_size`: the primary, then its
    Group members, then the rest of the subsystem by numeric channel proximity (nearby
    channel ids are physically adjacent parameters). Always within one subsystem. The
    ACTIVE subset actually shown to ChatTS is chosen later by `most_active_channels`.
    Pure — unit-testable."""
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
    kept = [r["Channel"] for r in ranked[:pool_size]]
    return sorted(kept, key=_channel_number)


def most_active_channels(
    series: dict[str, np.ndarray],
    centers: dict[str, int],
    primary: str,
    k: int = config.LABEL_MAX_SERIES,
    half: int = config.LABEL_COARSE_HALF,
) -> list[str]:
    """The primary + the (k-1) siblings with the largest std in the coarse window.

    ChatTS degrades on many channels, and near-flat siblings add noise, not signal —
    so only the channels that actually MOVE around the point are shown. Pure."""
    def activity(ch: str) -> float:
        c = centers[ch]
        w = series[ch][max(0, c - half) : c + half]
        return float(w.std()) if w.size else 0.0

    others = sorted((ch for ch in series if ch != primary), key=activity, reverse=True)
    kept = [primary, *others[: max(0, k - 1)]]
    return sorted(kept, key=_channel_number)


class ChannelCache:
    """Loads each ESA training channel once, memory-leanly (data as float32 + epoch
    timestamps). spaceai's per-channel load_and_preprocess expands a ~7M-row pickle and
    resamples it — big transient objects; we drop the ESA object and force a GC right
    after keeping only the two arrays, so the transients never accumulate across the
    (≤15) channels of a subsystem. `max_cached` bounds the resident set across points."""

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
            data = np.ascontiguousarray(esa.data[:, 0], dtype=np.float32)  # float32 halves resident memory
            times = np.asarray(esa.timestamps).astype("datetime64[s]").astype(np.int64)
            if len(self._store) >= self.max_cached:  # bound the cache — evict oldest
                self._store.pop(next(iter(self._store)))
            self._store[channel] = (data, times)
            del esa  # drop spaceai's DataFrames + torch tensors ...
            gc.collect()  # ... and release them now, before loading the next channel
            logger.info("loaded %s (%d samples, %.0f MB)", channel, times.size, (data.nbytes + times.nbytes) / 1e6)
        return self._store[channel]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Describe OOL points with ChatTS — whole subsystem, three granularities")
    parser.add_argument("--points", default=str(OUT_DIR / "ool_points.csv"), help="CSV from run_ool")
    parser.add_argument("--out", default=str(OUT_DIR / "descriptions.csv"))
    parser.add_argument("--root", default="space-ai/datasets", help="spaceai dataset root")
    parser.add_argument("--channel", default=None, help="restrict to one primary channel")
    parser.add_argument("--limit", type=int, default=None, help="describe at most N points (smoke runs)")
    parser.add_argument("--images-dir", default=str(OUT_DIR / "images"), help="where the per-point coarse/medium/fine PNGs go")
    parser.add_argument("--no-images", action="store_true", help="skip saving the PNGs")
    parser.add_argument("--backbone", choices=["vlm", "chatts"], default="vlm",
                        help="vlm = Groq vision LM on rendered plots (default); chatts = local ChatTS on raw values")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("transformers", "accelerate", "urllib3", "httpx", "groq"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    load_dotenv()  # GROQ_API_KEY(2/3) for the VLM backbone

    points = pd.read_csv(args.points, parse_dates=["timestamp"])
    if args.channel:
        points = points[points["channel"] == args.channel]
    if args.limit is not None:
        points = points.head(args.limit)
    if points.empty:
        logger.info("no OOL points to describe (%s)", args.points)
        return 0

    table = pd.read_csv(CHANNELS_CSV)
    cache = ChannelCache(args.root)
    descriptor: TimeSeriesDescriptor
    if args.backbone == "vlm":
        from src.labelling.vlm import GroqVLM  # noqa: PLC0415

        descriptor = VLMTimeSeriesDescriptor(GroqVLM())  # plots → Groq vision LM (fast, no 16 GB)
        logger.info("backbone: VLM %s (batches of ≤%d subplots)", config.GROQ_VLM_MODEL, config.VLM_BATCH_SIZE)
    else:
        from src.agentic.agents.perception import ChatTSPerceptor  # noqa: PLC0415

        descriptor = TimeSeriesDescriptor(ChatTSPerceptor())  # one shared 16 GB backend
        logger.info("backbone: ChatTS (local, MPS)")

    for _, row in points.iterrows():
        primary = str(row["channel"])
        instant = float(pd.Timestamp(row["timestamp"]).timestamp())
        pool = select_subsystem_channels(primary, table)  # same-subsystem pool (≤ LABEL_POOL_SIZE)

        pool_series: dict[str, np.ndarray] = {}
        pool_centers: dict[str, int] = {}
        for ch in pool:
            x, times = cache.get(ch)
            pool_series[ch] = x
            pool_centers[ch] = int(np.searchsorted(times, instant))  # time-aligned, per-channel index
        # keep only the primary + the few MOST ACTIVE siblings around the point
        active = most_active_channels(pool_series, pool_centers, primary)
        series = {ch: pool_series[ch] for ch in active}
        centers = {ch: pool_centers[ch] for ch in active}
        point = SubsystemPoint(primary=primary, centers=centers, label=int(row["label"]))
        logger.info(
            "%s @ %s: %d-channel pool → describing with %d most-active: %s",
            primary, row["timestamp"], len(pool), len(active), active,
        )
        descriptor.run(
            series, [point], out_path=args.out,
            images_dir=None if args.no_images else args.images_dir,
        )

    logger.info("descriptions saved → %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
