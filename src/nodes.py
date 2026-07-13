"""Deterministic graph nodes & helpers — pure functions, no LLM (§7.6, §11.2).

The non-reasoning pieces the loop runs: `preprocess`, canonical `channel_segments`
(cached — the analyst's tools use it), and `report`. There is NO binding verifier any
more: quantitative confirmation is a targeted analyst question and qualitative
confirmation a perceptor question (both non-binding), and the ORCHESTRATOR decides when
to conclude (ACCEPT). The old verifier/adjudication/belief-reallocation lives in legacy/.

Must not import from src.llm or src.agents.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from src import data
from src.models import AgentState, find_hypothesis
from src.tools import Segment, adaptive_pelt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical segmentation — cached, deterministic. Computed ONCE per (path, channel)
# and reused by the analyst's tools.
# ─────────────────────────────────────────────────────────────────────────────
@functools.cache
def _channel_segments(data_path: str, channel: str) -> tuple[Segment, ...]:
    return tuple(adaptive_pelt(data.get_channel(data_path, channel)))


def channel_segments(data_path: str, channel: str) -> list[Segment]:
    return list(_channel_segments(data_path, channel))


# ─────────────────────────────────────────────────────────────────────────────
# preprocess
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(state: AgentState) -> dict[str, Any]:
    """Initialise channels, the global range, and the loop control fields."""
    length = data.series_length(state["data"])
    channels = state.get("channels") or data.list_channels(state["data"])
    global_range = state.get("global_query_range") or (0, length)
    logger.info("preprocess: channels=%s length=%d range=%s", channels, length, global_range)
    return {
        "channels": channels,
        "global_query_range": global_range,
        "hypotheses": [],
        "orchestrator_iterations": 0,
        "current_action": None,
        "report": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# report — synthesise the evidence log and hypotheses (INV-3).
# The conclusion is what the orchestrator ACCEPTED (or, if it ran out of budget, the
# current top-plausibility hypothesis). Plausibilities are the LLM's own, set/refined at
# each (re)hypothesize; there is no deterministic reallocation.
# ─────────────────────────────────────────────────────────────────────────────
def report(state: AgentState) -> dict[str, Any]:
    action = state.get("current_action")
    accepted = action is not None and action.action == "conclude"

    ranked = sorted(state["hypotheses"], key=lambda h: h.plausibility, reverse=True)
    open_h = [h for h in ranked if h.status == "open"]
    top = None
    if accepted and action is not None and action.hypothesis_id:
        top = find_hypothesis(ranked, action.hypothesis_id)
    if top is None:
        top = open_h[0] if open_h else (ranked[0] if ranked else None)

    report_obj = {
        "concluded": bool(accepted),  # the orchestrator ACCEPTED (vs hitting the iteration budget)
        "n_iterations": state["orchestrator_iterations"],
        "conclusion": top.model_dump() if top is not None else None,
        "conclusion_reasoning": (
            action.reasoning if accepted and action is not None else "iteration budget reached"
        ),
        "ranking": [
            {"id": h.id, "type": h.type, "plausibility": round(h.plausibility, 3), "status": h.status}
            for h in ranked
        ],
        "n_evidence": len(state["evidence_log"]),
    }
    logger.info(
        "report: concluded=%s iterations=%d conclusion=%s",
        report_obj["concluded"], state["orchestrator_iterations"], top.id if top else None,
    )
    return {"report": report_obj}
