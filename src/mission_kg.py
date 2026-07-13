"""MissionKnowledgeGraph — a KG mined from ALL ESA-Mission1 CSVs.

Unlike `StubKnowledgeGraph` (channels.csv only — anti-leakage-strict), this KG
DELIBERATELY joins channels.csv + labels.csv + anomaly_types.csv to extract the
mission's accumulated operational knowledge, so the orchestrator can consult it
BEFORE spending a query on the analyst/perceptor:

  · channel hierarchy         (channels.csv: Subsystem / Physical Unit / Group / Target)
  · data-driven coupling      (labels.csv: which channels fail together)
  · per-subsystem anomaly priors  (anomaly_types.csv joined via labels:
                                   Class / Category / Dimensionality / Locality / Length)
  · mission-wide overview     (global base-rates: how anomalies look on THIS mission)

⚠️ labels.csv and anomaly_types.csv are GROUND TRUTH. To keep an evaluation honest,
construct the KG with `exclude_anomaly_ids={the id under test}` (leave-one-out):
the priors + co-occurrence + overview then reflect only OTHER anomalies, as an
engineer's prior experience would — never the answer to the window investigated (§0.5).

Implements the KnowledgeGraph Protocol; must not import src.llm / src.agents.
"""

from __future__ import annotations

import functools
from collections import Counter, defaultdict
from pathlib import Path
from typing import get_args

import pandas as pd

from src.models import HypothesisType

MISSION1_DIR = "space-ai/datasets/ESA-Mission1/ESA-Mission1"
GENERIC_VOCABULARY: list[str] = list(get_args(HypothesisType))
# The anomaly-type columns mined into per-subsystem priors (categorical fields).
_TYPE_COLUMNS = ("Class", "Category", "Dimensionality", "Locality", "Length")


@functools.cache
def _load_csvs(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(dataset_dir)
    return (
        pd.read_csv(d / "channels.csv"),
        pd.read_csv(d / "labels.csv"),
        pd.read_csv(d / "anomaly_types.csv"),
    )


def _top(counter: dict[str, int], k: int = 3) -> str:
    """Compact 'A×n, B×m' rendering of the k most frequent categories."""
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"{name}×{n}" for name, n in items) if items else "—"


class MissionKnowledgeGraph:
    """Rich KG over channels + labels + anomaly types (with leave-one-out support)."""

    def __init__(self, dataset_dir: str = MISSION1_DIR, exclude_anomaly_ids: frozenset[str] = frozenset()) -> None:
        channels, labels, anomaly_types = _load_csvs(dataset_dir)
        labels = labels[~labels["ID"].isin(exclude_anomaly_ids)]
        anomaly_types = anomaly_types[~anomaly_types["ID"].isin(exclude_anomaly_ids)]

        # channel hierarchy (channels.csv)
        self._sub: dict[str, str] = dict(zip(channels["Channel"], channels["Subsystem"], strict=True))
        self._unit: dict[str, str] = dict(zip(channels["Channel"], channels["Physical Unit"], strict=True))
        self._group: dict[str, str] = dict(zip(channels["Channel"], channels["Group"].astype(str), strict=True))
        self._target: dict[str, bool] = {
            ch: str(t).strip().upper() == "YES" for ch, t in zip(channels["Channel"], channels["Target"], strict=True)
        }
        self._by_unit: dict[str, list[str]] = defaultdict(list)
        for ch, unit in self._unit.items():
            self._by_unit[unit].append(ch)

        # data-driven co-occurrence: channels sharing an anomaly ID (labels.csv)
        self._cooc: dict[str, Counter[str]] = defaultdict(Counter)
        for _id, grp in labels.groupby("ID"):
            chans = sorted(set(grp["Channel"]))
            for i, a in enumerate(chans):
                for b in chans[i + 1 :]:
                    self._cooc[a][b] += 1
                    self._cooc[b][a] += 1

        # per-subsystem anomaly-type priors (one row per (id, subsystem))
        joined = labels.merge(channels[["Channel", "Subsystem"]], on="Channel", how="left").merge(
            anomaly_types, on="ID", how="left"
        )
        per_sub = joined.drop_duplicates(["ID", "Subsystem"])
        self._priors: dict[str, dict[str, dict[str, int]]] = {}
        for sub, g in per_sub.groupby("Subsystem"):
            self._priors[str(sub)] = {
                col: {str(k): int(v) for k, v in g[col].value_counts().items()} for col in _TYPE_COLUMNS
            }

        # mission-wide base-rates over the anomalies still in scope (leave-one-out aware)
        self._n_anomalies = int(anomaly_types["ID"].nunique())
        self._overview: dict[str, dict[str, int]] = {
            col: {str(k): int(v) for k, v in anomaly_types[col].value_counts().items()} for col in _TYPE_COLUMNS
        }

    # ── KnowledgeGraph Protocol ──────────────────────────────────────────────
    def related_channels(self, ch: str, k: int = 8) -> list[str]:
        """Channels that historically fail WITH `ch` (data-driven), else same unit."""
        if ch in self._cooc and self._cooc[ch]:
            return [c for c, _ in self._cooc[ch].most_common(k)]
        unit = self._unit.get(ch)
        return [c for c in self._by_unit.get(unit, []) if c != ch] if unit else []

    def subsystem_of(self, ch: str) -> str:
        return self._sub.get(ch, "unknown")

    def hypothesis_vocabulary(self) -> list[str]:
        return list(GENERIC_VOCABULARY)

    def hypothesis_prior(self, channels: list[str]) -> str:
        """A NL prior on what KIND of anomaly to expect for these channels' subsystem
        — biases hypothesis formulation (Phase 5: the KG informing `hypothesize`)."""
        subs = {self.subsystem_of(c) for c in channels if self.subsystem_of(c) != "unknown"}
        lines: list[str] = []
        for sub in sorted(subs):
            p = self._priors.get(sub)
            if not p:
                continue
            dim, loc, length = p["Dimensionality"], p["Locality"], p["Length"]
            multivariate = dim.get("Multivariate", 0) >= dim.get("Univariate", 0)
            mostly_global = loc.get("Global", 0) >= loc.get("Local", 0)
            pointwise = length.get("Point", 0) > length.get("Subsequence", 0)
            hint = (
                "usually MULTIVARIATE (coupled channels) → favour cross_channel_coupling / regime_change"
                if multivariate
                else "usually UNIVARIATE → favour level_shift / drift / variance_change"
            )
            scope = "global regime changes" if mostly_global else "LOCAL sub-interval events (zoom in to localise)"
            length_hint = "brief point spikes" if pointwise else "extended subsequences"
            lines.append(
                f"  {sub}: historically {hint}; {scope}; {length_hint}. "
                f"(Class={_top(p['Class'])}; Category={_top(p['Category'])})"
            )
        return ("Mission anomaly priors for these channels' subsystem(s):\n" + "\n".join(lines) + "\n\n") if lines else ""

    # ── extras (not in the Protocol) ─────────────────────────────────────────
    def physical_unit_of(self, ch: str) -> str:
        return self._unit.get(ch, "unknown")

    def is_target(self, ch: str) -> bool:
        """Whether `ch` is a monitored TARGET channel (channels.csv Target=YES)."""
        return self._target.get(ch, False)

    def anomaly_profile(self, subsystem: str) -> dict[str, dict[str, int]]:
        return self._priors.get(subsystem, {})

    def mission_overview(self) -> str:
        """Mission-wide base-rates the orchestrator should read BEFORE its first query:
        how anomalies look on THIS mission (a prior, before touching the window)."""
        o = self._overview
        dim = o["Dimensionality"]
        multi = dim.get("Multivariate", 0)
        total_dim = multi + dim.get("Univariate", 0)
        pct = round(100 * multi / total_dim) if total_dim else 0
        return (
            f"Mission base-rates over {self._n_anomalies} known anomalies (prior experience):\n"
            f"  · Dimensionality: {pct}% MULTIVARIATE → a lone channel rarely tells the whole story; "
            f"check coupled channels early.\n"
            f"  · Locality: {_top(o['Locality'], 2)} → many events are LOCAL, invisible at whole-window scale; "
            f"zoom into flagged sub-intervals.\n"
            f"  · Length: {_top(o['Length'], 2)} → mostly extended subsequences, some single points.\n"
            f"  · Category: {_top(o['Category'], 3)}.\n\n"
        )
