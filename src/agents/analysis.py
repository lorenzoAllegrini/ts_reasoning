"""Analysis agent — the QUANTITATIVE analyst, an independent tool-using sub-agent.

Given a question + coordinates from the orchestrator, it DECIDES which statistical
tools to run (a structured decision — its "ToolNode", implemented via json_schema
because gpt-oss-20b's native tool-calling is unreliable), runs them deterministically,
and SYNTHESISES a quantitative description. It DESCRIBES; it never issues a verdict
(only the deterministic verifier closes — INV-1). Its output is non-binding evidence.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from src import config, data
from src.agents.base import BaseLLMAgent
from src.nodes import channel_segments
from src.tools import (
    compare_intervals_statistics,
    compute_cross_channel_coupling,
    compute_interval_anomaly_score,
    point_out_of_limits,
)

logger = logging.getLogger(__name__)

AnalystTool = Literal[
    "segment", "compare_intervals", "anomaly_score", "cross_channel_coupling", "out_of_limits"
]

TOOLBOX = {
    "segment": "adaptive PELT change-point segmentation of a channel (where regimes start/end)",
    "compare_intervals": "Mann-Whitney U, KS and Cohen's d of query vs context on a channel",
    "anomaly_score": "per-metric breakdown query vs context (mean_deviation, trend_divergence, "
    "volatility_collapse, volatility_change, distribution_shift) + dominant metric",
    "cross_channel_coupling": "lead/lag cross-correlation between two channels",
    "out_of_limits": "targeted point check: is any query point decidedly OUT OF LIMITS "
    "(beyond context mean ± k·std)? reports the most-extreme point and how many σ out",
}

SELECT_SYSTEM_PROMPT = f"""\
You are the STATISTICAL ANALYST for ESA spacecraft telemetry, deciding which tools
to run to answer a question about a query interval vs a context interval. Choose the
minimal set of tools that answers it. Your toolbox:
{chr(10).join(f"  - {name}: {desc}" for name, desc in TOOLBOX.items())}
Use cross_channel_coupling only when two channels are given. Return a list of tool
invocations (tool + channel, and channel_b for coupling)."""

SYNTH_SYSTEM_PROMPT = """\
You are the STATISTICAL ANALYST. Given the numbers your tools produced, write a short
QUANTITATIVE description that answers the question. Describe what the statistics show
(level shifts, trends, volatility, coupling), citing the numbers. You DESCRIBE — you do
NOT decide whether a hypothesis is confirmed, whether something is an anomaly, or assign
any verdict. Two or three sentences. No verdict, no recommendation."""


class AnalystToolInvocation(BaseModel):
    tool: AnalystTool
    channel: str
    channel_b: str = ""  # only for cross_channel_coupling


class AnalystToolPlan(BaseModel):
    invocations: list[AnalystToolInvocation]


class AnalysisAgent(BaseLLMAgent):
    """Tool-using statistical analyst. describe() = select tools → run → synthesise."""

    def _run_tool(
        self, data_path: str, inv: AnalystToolInvocation, q: tuple[int, int], c: tuple[int, int]
    ) -> str:
        try:
            if inv.tool == "segment":
                segs = channel_segments(data_path, inv.channel)
                inside = [s for s in segs if q[0] <= s["start"] < q[1] or q[0] < s["end"] <= q[1]]
                bounds = [s["start"] for s in segs if 0 < s["start"]]
                return (
                    f"segment[{inv.channel}]: {len(segs)} regimes, change-points at {bounds[:8]}; "
                    f"{len(inside)} boundary/-ies within {q}"
                )
            if inv.tool == "compare_intervals":
                cmp_ = compare_intervals_statistics(data.get_channel(data_path, inv.channel), q, c)
                return (
                    f"compare[{inv.channel}] {q} vs {c}: cohens_d={cmp_['cohens_d']:.2f}, "
                    f"ks_p={cmp_['ks_p']:.2g}, mean {cmp_['mean_query']:.2f} vs {cmp_['mean_context']:.2f}"
                )
            if inv.tool == "anomaly_score":
                score = compute_interval_anomaly_score(data.get_channel(data_path, inv.channel), q, c)
                metrics = {k: round(v, 2) for k, v in score["metrics"].items()}
                return (
                    f"anomaly_score[{inv.channel}] {q} vs {c}: dominant={score['dominant_metric']}, "
                    f"composite={score['composite']:.2f}, metrics={metrics}"
                )
            if inv.tool == "cross_channel_coupling":
                b = inv.channel_b or inv.channel
                cp = compute_cross_channel_coupling(
                    data.get_channel(data_path, inv.channel), data.get_channel(data_path, b), q
                )
                return f"coupling[{inv.channel},{b}] {q}: peak_corr={cp['peak_corr']:.2f}, lag={cp['lag']}"
            if inv.tool == "out_of_limits":
                ool = point_out_of_limits(data.get_channel(data_path, inv.channel), q, c)
                where = f"at idx {ool['extreme_index']}" if ool["out_of_limits"] else "none out"
                return (
                    f"out_of_limits[{inv.channel}] {q} vs {c}: {ool['n_out']} point(s) beyond "
                    f"±{ool['limit_sigma']:.0f}σ (max |z|={ool['max_abs_z']:.1f}, {where})"
                )
        except (KeyError, ValueError) as exc:
            return f"({inv.tool}[{inv.channel}] failed: {exc})"
        return f"(unknown tool {inv.tool})"

    def describe(
        self,
        data_path: str,
        question: str,
        channels: list[str],
        query_range: tuple[int, int],
        context_range: tuple[int, int],
    ) -> str:
        """Answer the orchestrator's question with a quantitative description."""
        select_human = (
            f"Channels available: {channels}. Query interval: {query_range}. "
            f"Context interval: {context_range}.\nQuestion: {question}\n"
            "Which tools should we run?"
        )
        plan = self.structured(AnalystToolPlan, SELECT_SYSTEM_PROMPT, select_human)
        assert isinstance(plan, AnalystToolPlan)
        results = [
            self._run_tool(data_path, inv, query_range, context_range)
            for inv in plan.invocations[: config.ANALYST_MAX_TOOLS]
        ]
        if not results:  # the analyst chose nothing → give it the default comparison
            results = [
                self._run_tool(
                    data_path, AnalystToolInvocation(tool="anomaly_score", channel=ch), query_range, context_range
                )
                for ch in channels[:2]
            ]
        synth_human = f"Question: {question}\nTool results:\n" + "\n".join(f"  {r}" for r in results)
        description = self.text(SYNTH_SYSTEM_PROMPT, synth_human)
        logger.info("analyst: %s", description[:160])
        return description
