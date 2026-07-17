"""Typed data contract — AgentState, Hypothesis, EvidenceEntry (§4).

The state carries a *set* of hypotheses with plausibilities (not a single string):
refinement IS the evolution of that set. The evidence log is append-only and is
the deliverable (INV-3). Every entry records `source` (perception | deterministic)
so INV-1 can be enforced in code: perceptual entries can never close a hypothesis.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ── Closed vocabularies ──────────────────────────────────────────────────────
HypothesisType = Literal[
    "level_shift",
    "regime_change",
    "drift",
    "variance_change",
    "sensor_fault",
    "cross_channel_coupling",  # multivariate: two channels move together / one leads
    "novel",
]
Verdict = Literal["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "OBSERVATION"]
# Epistemic gate (INV-1): only "deterministic" entries can carry a closing verdict.
# "analysis" = the statistical sub-agent's quantitative DESCRIPTION (non-binding);
# "perception" = the perceptor's qualitative OBSERVATION (non-binding).
Source = Literal["perception", "deterministic", "analysis"]
Status = Literal["open", "supported", "eliminated"]
ProbeMode = Literal[
    "describe", "compare", "feature", "cross_channel", "localisation", "discrimination"
]
# The orchestrator's move each iteration. It MAY "conclude" (ACCEPT) when its targeted
# questions have made it confident — stopping is now the orchestrator's call, with the
# iteration budget as a backstop. There is no binding "verify" action any more.
OrchestratorActionType = Literal["describe", "perceive", "hypothesize", "kg_lookup", "conclude"]


class Hypothesis(BaseModel):
    """A competing, mechanism-grounded explanation with a plausibility."""

    id: str
    type: HypothesisType
    description: str  # NL, often originated from perception
    predicted_signature: str  # what evidence would CONFIRM it
    refuting_evidence: str  # what evidence would KILL it — the key to the PLAN
    plausibility: float  # sums to 1 over open hypotheses
    status: Status = "open"


class EvidenceEntry(BaseModel):
    """One row of the audit trail. `source` is the epistemic gate (INV-1)."""

    step: int
    hypothesis_id: str  # "" for perceptual observations (hypothesis-independent)
    source: Source
    action: str  # tool/probe name
    channel: str  # single name, or "+"-joined for multivariate / cross-channel
    query_range: tuple[int, int]
    context_range: tuple[int, int]
    raw_result: dict[str, Any]  # raw statistics
    verdict: Verdict
    plausibility_delta: dict[str, float]  # {hyp_id: delta}
    rationale: str


class ProbeRequest(BaseModel):
    """Orchestrator → Perceptor. Deliberately carries NO hypothesis or plausibility.

    If the perceptor knew which hypotheses are in play it would tend to see what is
    needed, and its observation would stop being independent evidence. The isolation
    is guaranteed by the TYPE, not the prompt — the probe node reads only this.
    """

    mode: ProbeMode
    question: str
    channels: list[str]
    query_range: tuple[int, int]
    reference_range: tuple[int, int] | None = None


class OrchestratorAction(BaseModel):
    """The orchestrator's chosen move — a structured decision the dispatcher runs.

    One fat-but-flat schema (robust with json_schema on the weak model). Only the
    fields relevant to `action` are used: describe/perceive/verify use the ranges +
    channels; describe/perceive use `question`; verify uses `hypothesis_id`;
    kg_lookup uses `kg_channel`.
    """

    action: OrchestratorActionType
    # ONE short sentence. A long free-text reasoning overruns the weak model's
    # constrained-decoding budget and the JSON never closes (Groq json_validate_failed);
    # it also re-enters every later `decide` prompt, so brevity bounds the token cost too.
    reasoning: str = Field(description="One short sentence (≤25 words, single line) — why this move discriminates.")
    channels: list[str] = []
    query_start: int = 0
    query_end: int = 0
    context_start: int = 0
    context_end: int = 0
    question: str = ""  # for describe / perceive
    probe_mode: ProbeMode = "describe"  # for perceive
    hypothesis_id: str = ""  # for conclude (names the hypothesis being ACCEPTed as the answer)
    kg_channel: str = ""  # for kg_lookup

    @property
    def query_range(self) -> tuple[int, int]:
        return (self.query_start, self.query_end)

    @property
    def context_range(self) -> tuple[int, int]:
        return (self.context_start, self.context_end)


class AgentState(TypedDict):
    """LangGraph state for the agentic loop. Nodes return PARTIAL updates.

    The orchestrator drives an investigation freely (describe/perceive/verify/…);
    the evidence log holds EVERY move (descriptions, observations, verdicts) — it is
    the deliverable (INV-3). Belief and termination stay deterministic.
    """

    # --- data ---
    data: str  # identifier/path resolved to channel arrays by src/data.py
    channels: list[str]
    user_question: str
    global_query_range: tuple[int, int]

    # --- CORE: hypothesis state ---
    hypotheses: list[Hypothesis]

    # --- evidence log (append-only, reducer = operator.add) — descriptions,
    #     observations AND verdicts; the full investigation trace ---
    evidence_log: Annotated[list[EvidenceEntry], operator.add]

    # --- agentic loop control ---
    current_action: OrchestratorAction | None  # the move just chosen (routing); None → budget stop
    orchestrator_iterations: int  # total moves so far (hard budget → report)

    # --- final synthesis (written by the report node) ---
    report: dict[str, Any] | None


def find_hypothesis(hypotheses: list[Hypothesis], hyp_id: str) -> Hypothesis | None:
    """Return the hypothesis with this id, or None."""
    return next((h for h in hypotheses if h.id == hyp_id), None)
