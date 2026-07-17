"""Knowledge graph — interface + stub over the ESA-ADB channel hierarchy.

Kept as its OWN module (not merged into data.py) precisely to preserve the
anti-leakage guarantee: this file touches channels.csv ONLY. The real mission
ontology/KG is future work (Phase 5); the stub exposes the REAL hierarchy already
in channels.csv (Subsystem → Physical Unit → Group → Channel), which the plan node
uses to pass PHYSICALLY-COUPLED channels together (§0.5).

⚠️ Anti-leakage (non-negotiable): channels.csv is METADATA and may be used freely.
Nothing here reads labels.csv or the event types in anomaly_types.csv — those are
ground truth (enforced by tests/test_kg.py). PURE/deterministic: no src.llm/src.agents.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Protocol, get_args, runtime_checkable

import pandas as pd

from src import config
from src.agentic.models import HypothesisType

# Generic hypothesis vocabulary available WITHOUT a mission ontology (§0.5):
# surface mechanisms, not named physical faults. Kept in sync with HypothesisType.
GENERIC_VOCABULARY: list[str] = list(get_args(HypothesisType))


@runtime_checkable
class KnowledgeGraph(Protocol):
    def related_channels(self, ch: str) -> list[str]:
        """Channels physically coupled to `ch` (same physical unit)."""
        ...

    def subsystem_of(self, ch: str) -> str:
        """The subsystem `ch` belongs to (or 'unknown')."""
        ...

    def hypothesis_vocabulary(self) -> list[str]:
        """The mechanisms the orchestrator may hypothesise for this mission."""
        ...


@functools.cache
def _load_hierarchy(channels_csv: str) -> tuple[dict[str, tuple[str, str]], dict[str, list[str]]]:
    """Return (channel -> (physical_unit, subsystem), physical_unit -> [channels])."""
    by_channel: dict[str, tuple[str, str]] = {}
    by_unit: dict[str, list[str]] = {}
    path = Path(channels_csv)
    if not path.exists():
        return by_channel, by_unit
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        channel = str(row["Channel"])
        unit = str(row["Physical Unit"])
        subsystem = str(row["Subsystem"])
        by_channel[channel] = (unit, subsystem)
        by_unit.setdefault(unit, []).append(channel)
    return by_channel, by_unit


class StubKnowledgeGraph:
    """Channel hierarchy from channels.csv (metadata only)."""

    def __init__(self, channels_csv: str = config.CHANNELS_CSV) -> None:
        self._by_channel, self._by_unit = _load_hierarchy(channels_csv)

    def related_channels(self, ch: str) -> list[str]:
        info = self._by_channel.get(ch)
        if info is None:
            return []
        unit, _ = info
        return [c for c in self._by_unit.get(unit, []) if c != ch]

    def subsystem_of(self, ch: str) -> str:
        info = self._by_channel.get(ch)
        return info[1] if info is not None else "unknown"

    def hypothesis_vocabulary(self) -> list[str]:
        return list(GENERIC_VOCABULARY)
