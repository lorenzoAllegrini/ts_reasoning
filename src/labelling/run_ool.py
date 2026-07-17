"""OOL detection over the ESA-Mission1 TRAINING telemetry — rule-based, no LLM.

    uv run python -m src.labelling.run_ool                      # benchmark target channels
    uv run python -m src.labelling.run_ool --channels channel_41 channel_42
    uv run python -m src.labelling.run_ool --all                # every parameter channel

Channels are loaded and preprocessed by spaceai's ESA loader (space-ai/spaceai/data/
esa.py: 30 s zero-order-hold resampling, gap-aware blocks, train = 2000→2007). Each
channel is scanned with strided rolling windows over three features (median / std /
zero-crossing rate); candidates clear a hard robust-z floor. Selection is PER CHANNEL
and balanced PER SUBSYSTEM (see ool.select_balanced): every subsystem gets an equal
share of the budget, filled round-robin over its channels by each channel's OWN
z-ranking — z values are never compared across channels — with the ≥ 1-day distinct-
points guarantee kept global. The grand total stays strictly below
config.OOL_MAX_POINTS. Ground-truth labels ride along for EVALUATION only (they never
influence detection).

Output: labelling_out/ool_points.csv  (channel, center, timestamp, feature, z, label)
        labelling_out/ool_meta.json   (parameters, per-channel thresholds, per-subsystem counts)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.labelling.ool import OOLCandidate, channel_candidates, select_balanced

logger = logging.getLogger(__name__)

OUT_DIR = Path("labelling_out")


def subsystem_map() -> dict[str, str]:
    """channel → subsystem, from the mission metadata (channels.csv)."""
    table = pd.read_csv(config.CHANNELS_CSV)
    return dict(zip(table["Channel"], table["Subsystem"], strict=True))


def _epoch_seconds(timestamps: np.ndarray) -> np.ndarray:
    return timestamps.astype("datetime64[s]").astype(np.float64)


def collect_candidates(root: str, channels: list[str]) -> tuple[list[OOLCandidate], dict[str, int]]:
    """Load each training channel via spaceai's preprocessing and scan it for OOL windows."""
    from spaceai.data.esa import ESA, ESAMissions  # noqa: PLC0415 — torch-heavy, script-only

    mission = ESAMissions.MISSION_1.value
    all_candidates: list[OOLCandidate] = []
    sizes: dict[str, int] = {}
    for i, channel in enumerate(channels):
        t0 = time.time()
        esa = ESA(root=root, mission=mission, channel_id=channel, train=True, download=False)
        x = esa.data[:, 0].astype(np.float64)
        cands = channel_candidates(
            x, channel, _epoch_seconds(np.asarray(esa.timestamps)), esa.anomalies
        )
        all_candidates.extend(cands)
        sizes[channel] = int(x.size)
        logger.info(
            "[%d/%d] %s: %d samples, %d anomaly intervals, %d OOL candidates (%.1fs)",
            i + 1, len(channels), channel, x.size, len(esa.anomalies), len(cands), time.time() - t0,
        )
    return all_candidates, sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rule-based OOL detection on ESA training telemetry")
    parser.add_argument("--root", default="space-ai/datasets", help="spaceai dataset root")
    parser.add_argument("--channels", nargs="*", default=None, help="explicit channel ids")
    parser.add_argument("--all", action="store_true", help="all 76 parameter channels (default: the labelled benchmark target channels)")
    parser.add_argument("--out", default=str(OUT_DIR / "ool_points.csv"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from spaceai.data.esa import ESAMissions  # noqa: PLC0415

    mission = ESAMissions.MISSION_1.value
    channels = args.channels or (mission.parameters if args.all else mission.target_channels)

    candidates, sizes = collect_candidates(args.root, channels)
    subsystems = subsystem_map()
    selected, per_channel_threshold, per_subsystem = select_balanced(candidates, subsystems)

    frame = pd.DataFrame(
        {
            "channel": [c["channel"] for c in selected],
            "subsystem": [subsystems.get(c["channel"], "unknown") for c in selected],
            "center": [c["center"] for c in selected],
            "timestamp": [pd.Timestamp(c["time_s"], unit="s") for c in selected],
            "feature": [c["feature"] for c in selected],
            "z": [round(c["z"], 3) for c in selected],
            "label": [c["label"] for c in selected],
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    meta = {
        "n_channels": len(channels),
        "n_candidates_above_floor": len(candidates),
        "n_selected": len(selected),
        "z_floor": config.OOL_Z_FLOOR,
        "window": config.OOL_WINDOW,
        "hop": config.OOL_HOP,
        "min_separation_s": config.OOL_MIN_SEPARATION_S,
        "max_points_strict_upper_bound": config.OOL_MAX_POINTS,
        "per_subsystem": per_subsystem,
        "per_channel_threshold": {ch: round(z, 3) for ch, z in sorted(per_channel_threshold.items())},
        "per_feature": frame["feature"].value_counts().to_dict() if len(frame) else {},
        "label_hit_rate": float(frame["label"].mean()) if len(frame) else None,
        "channel_sizes": sizes,
    }
    (OUT_DIR / "ool_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "OOL: %d points (< %d) from %d candidates → %s | per-subsystem %s | %d channels with own threshold | label hit rate %s",
        len(selected), config.OOL_MAX_POINTS, len(candidates), args.out,
        per_subsystem, len(per_channel_threshold), meta["label_hit_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
