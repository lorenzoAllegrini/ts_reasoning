"""TelemetryInterpretationGraph — the bounded-agent LangGraph, unified in one class.

The orchestrator is a tool-calling agent: it DECIDES the next move and a dispatcher runs
it. Every move is an INVESTIGATION — describe (analyst, quantitative), perceive (perceptor,
qualitative), hypothesize (refine the competing set), kg_lookup — asking increasingly
targeted questions. There is NO binding verifier: when the orchestrator's questions have
made it confident it emits `conclude` (ACCEPT). Stopping is the orchestrator's call, with
the iteration budget as a backstop; plausibilities are the orchestrator's own.

    START → preprocess → (seed: analyst ∥ perceptor) → orchestrate ⇄ {describe, perceive,
    hypothesize, kg_lookup};  conclude / budget → report → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src import config
from src.agentic.agents.analysis import AnalysisAgent
from src.agentic.agents.orchestrator import Orchestrator
from src.agentic.agents.perception import Perceptor
from src.agentic.kg import KnowledgeGraph, StubKnowledgeGraph
from src.agentic.llm import get_llms
from src.agentic.models import AgentState, EvidenceEntry, ProbeRequest
from src.agentic.nodes import preprocess, report

logger = logging.getLogger(__name__)

_SEED_QUESTION = "Characterise the whole window: where are the change-points and regimes, and what stands out?"


class TelemetryInterpretationGraph:
    """Owns the reasoning agents + the KG and the compiled bounded-agent workflow."""

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        perceptor: Perceptor | None = None,
        analyst: AnalysisAgent | None = None,
        kg: KnowledgeGraph | None = None,
    ) -> None:
        if orchestrator is None or analyst is None:
            llms = get_llms()  # primary + any fallback keys (GROQ_API_KEY2…)
            primary, fallbacks = llms[0], llms[1:]
            kg = kg or StubKnowledgeGraph()
            orchestrator = orchestrator or Orchestrator(primary, kg, fallbacks=fallbacks)
            analyst = analyst or AnalysisAgent(primary, fallbacks=fallbacks)
        self.orchestrator = orchestrator
        self.perceptor = perceptor or Perceptor()
        self.analyst = analyst
        self.kg = kg or StubKnowledgeGraph()
        self.graph = self._build_graph()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _range(self, rng: tuple[int, int], state: AgentState) -> tuple[int, int]:
        """Fall back to the global range if the LLM left a coordinate at (0, 0)."""
        return rng if rng != (0, 0) else state["global_query_range"]

    @staticmethod
    def _obs(
        step: int, source: Any, action: str, channels: list[str],
        q: tuple[int, int], c: tuple[int, int], text: str, raw: dict[str, Any],
    ) -> EvidenceEntry:
        return EvidenceEntry(
            step=step, hypothesis_id="", source=source, action=action,
            channel="+".join(channels) if channels else "-", query_range=q, context_range=c,
            raw_result=raw, verdict="OBSERVATION", plausibility_delta={}, rationale=text,
        )

    # ── seed: the initial interval → both scouts (§ "entrambi i riassunti") ──
    def _seed_analyst_node(self, state: AgentState) -> dict[str, Any]:
        g = state["global_query_range"]
        desc = self.analyst.describe(state["data"], _SEED_QUESTION, state["channels"], g, g)
        return {"evidence_log": [self._obs(0, "analysis", "seed_describe", state["channels"][:3], g, g, desc, {"description": desc})]}

    def _seed_perceptor_node(self, state: AgentState) -> dict[str, Any]:
        g = state["global_query_range"]
        req = ProbeRequest(mode="describe", question=_SEED_QUESTION, channels=state["channels"][:3], query_range=g)
        return {"evidence_log": [self.perceptor.probe(req, 0, state["data"])]}

    # ── orchestrate: decide the next move (or conclude on budget) ────────────
    def _orchestrate_node(self, state: AgentState) -> dict[str, Any]:
        if state["orchestrator_iterations"] >= config.MAX_ORCHESTRATOR_ITERATIONS:
            logger.info("orchestrate: iteration budget reached → conclude")
            return {"current_action": None}
        try:
            action = self.orchestrator.decide(
                state["hypotheses"], state["evidence_log"], state["channels"], state["global_query_range"]
            )
        except Exception as exc:  # noqa: BLE001 — a provider outage/rate-limit must not destroy the run
            logger.warning("orchestrate: decide failed (%s) → conclude with the evidence gathered so far", exc)
            return {"current_action": None}  # → report; the log is still the auditable deliverable (INV-3)
        return {"current_action": action, "orchestrator_iterations": state["orchestrator_iterations"] + 1}

    def _route_action(self, state: AgentState) -> str:
        action = state["current_action"]
        if action is None or action.action == "conclude":  # ACCEPT or budget → wrap up
            return "report"
        return {
            "describe": "run_describe", "perceive": "run_perceive",
            "hypothesize": "run_hypothesize", "kg_lookup": "run_kg",
        }[action.action]

    # ── investigation executors (loop back to orchestrate) ───────────────────
    def _describe_node(self, state: AgentState) -> dict[str, Any]:
        a = state["current_action"]
        assert a is not None
        channels = a.channels or state["channels"][:2]
        q, c = self._range(a.query_range, state), self._range(a.context_range, state)
        desc = self.analyst.describe(state["data"], a.question or "Describe this interval.", channels, q, c)
        step = state["orchestrator_iterations"]
        return {"evidence_log": [self._obs(step, "analysis", "describe", channels, q, c, desc, {"question": a.question, "description": desc})]}

    def _perceive_node(self, state: AgentState) -> dict[str, Any]:
        a = state["current_action"]
        assert a is not None
        channels = a.channels or state["channels"][:2]
        ref = a.context_range if a.context_range != (0, 0) else None
        req = ProbeRequest(mode=a.probe_mode, question=a.question or "describe what you see",
                           channels=channels, query_range=self._range(a.query_range, state), reference_range=ref)
        return {"evidence_log": [self.perceptor.probe(req, state["orchestrator_iterations"], state["data"])]}

    def _kg_node(self, state: AgentState) -> dict[str, Any]:
        a = state["current_action"]
        assert a is not None
        ch = a.kg_channel or (state["channels"][0] if state["channels"] else "")
        related = self.kg.related_channels(ch)
        info = f"KG: {ch} in {self.kg.subsystem_of(ch)}; physically coupled with {related}"
        step = state["orchestrator_iterations"]
        return {"evidence_log": [self._obs(step, "analysis", "kg_lookup", [ch], (0, 0), (0, 0), info, {"related": related})]}

    def _hypothesize_node(self, state: AgentState) -> dict[str, Any]:
        eliminated = [h for h in state["hypotheses"] if h.status == "eliminated"]
        hyps = self.orchestrator.hypothesize(state["evidence_log"], eliminated, state["channels"])
        return {"hypotheses": hyps}

    def _report_node(self, state: AgentState) -> dict[str, Any]:
        return report(state)

    # ── graph construction ───────────────────────────────────────────────────
    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        nodes = {
            "preprocess": lambda s: preprocess(s),
            "seed_analyst": self._seed_analyst_node,
            "seed_perceptor": self._seed_perceptor_node,
            "orchestrate": self._orchestrate_node,
            "run_describe": self._describe_node,
            "run_perceive": self._perceive_node,
            "run_kg": self._kg_node,
            "run_hypothesize": self._hypothesize_node,
            "report": self._report_node,
        }
        for name, method in nodes.items():
            g.add_node(name, method)  # type: ignore[call-overload]

        g.add_edge(START, "preprocess")
        g.add_edge("preprocess", "seed_analyst")
        g.add_edge("preprocess", "seed_perceptor")
        g.add_edge("seed_analyst", "orchestrate")
        g.add_edge("seed_perceptor", "orchestrate")
        g.add_conditional_edges(
            "orchestrate", self._route_action,
            {
                "run_describe": "run_describe", "run_perceive": "run_perceive",
                "run_hypothesize": "run_hypothesize", "run_kg": "run_kg", "report": "report",
            },
        )
        for executor in ("run_describe", "run_perceive", "run_hypothesize", "run_kg"):
            g.add_edge(executor, "orchestrate")  # investigation loops back
        g.add_edge("report", END)
        return g.compile()

    def run(self, state: AgentState) -> AgentState:
        result: AgentState = self.graph.invoke(state, config={"recursion_limit": config.RECURSION_LIMIT})
        return result


def make_initial_state(
    data_path: str,
    user_question: str,
    channels: list[str] | None = None,
    global_query_range: tuple[int, int] | None = None,
) -> AgentState:
    """A complete initial AgentState (all keys present so nodes never KeyError)."""
    return AgentState(
        data=data_path,
        channels=channels or [],
        user_question=user_question,
        global_query_range=global_query_range or (0, 0),
        hypotheses=[],
        evidence_log=[],
        current_action=None,
        orchestrator_iterations=0,
        report=None,
    )
