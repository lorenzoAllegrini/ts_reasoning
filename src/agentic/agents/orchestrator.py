"""Orchestrator agent — the SCIENTIST, a bounded tool-calling agent (§7.1).

It commands a team (statistical ANALYST, PERCEPTOR, KG) and DECIDES the next move each
iteration (`decide` → OrchestratorAction). It reasons about hypotheses and strategy by
asking increasingly targeted questions; it never computes a statistic itself. There is NO
binding verifier: quantitative confirmation is a targeted analyst question (e.g.
out_of_limits) and qualitative confirmation a perceptor question — both non-binding — and
the orchestrator itself decides when to `conclude` (ACCEPT). Plausibilities are its own,
assigned/refined at each (re)hypothesize.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from src.agentic.agents.base import BaseLLMAgent
from src.agentic.kg import KnowledgeGraph
from src.agentic.models import (
    Hypothesis,
    HypothesisType,
    OrchestratorAction,
)

logger = logging.getLogger(__name__)

# ── §7.1 role preamble (shared) ──────────────────────────────────────────────
# The evidence principles + the iterative refine-by-selection style are adapted from
# ARTIST (Messica et al., ICML 2026, "Adaptive Time Series Reasoning via Segment
# Selection"): reasoning interleaved with adaptive segment selection, where a value is
# informative only relative to a chosen baseline and selection must be sequential.
ORCHESTRATOR_PREAMBLE = """\
You are the ORCHESTRATOR of an iterative telemetry-interpretation loop (ESA-ADB). You
are the SCIENTIST: you formulate competing explanations, design discriminating
experiments, and interpret evidence. You do NOT compute statistics or decide verdicts.

YOUR TEAM — know exactly what each one is good for:
  • the ANALYST (statistical, QUANTITATIVE) — put a NUMBER on a query range vs a context:
    change-points/segments, Cohen's d + KS between two intervals, a composite anomaly
    score with its dominant metric, cross-channel peak-correlation + lead/lag. Ask it
    "how big, how significant, where is the change-point".
  • the PERCEPTOR (native TS-LLM, QUALITATIVE) — describes what a segment LOOKS LIKE in
    open vocabulary and, crucially, COMPARES SHAPES ACROSS RANGES. Ask it for:
      (1) holistic cross-range comparison — "does the pattern in [a,b] resemble [c,d]? how
          do they differ?" (its documented strength; often the move that unblocks a loop,
          e.g. revealing the discriminator is the MODULATION, not the level);
      (2) feature presence/absence — "is the periodic modulation present, attenuated, or
          absent here?";
      (3) cross-channel relation — "do ch_X and ch_Y move together, or does one lead?"
          (this mission is ~84% MULTIVARIATE, so this is high-value);
      (4) qualitative localisation — "is there a point where the shape changes character?";
      (5) shape discrimination — "does this look more like a saturation or a truncated step?";
      (6) description of the unnamed — first-encounter, open-ended "what is this?".
    The ANALYST and PERCEPTOR only DESCRIBE; neither can ever confirm or eliminate a hypothesis.
  • the VERIFIER — runs a deterministic test; its verdict is BINDING. ONLY it closes a
    hypothesis. You must NEVER overturn a verdict with your own reasoning.

CONSULT THE KG FIRST. Before every query you are given mission knowledge (base-rates,
per-subsystem anomaly priors, physically-coupled channels). You MUST reason FROM it before
spending a query: it tells you where to look and what to expect (e.g. "this subsystem's
anomalies are mostly LOCAL and multivariate → a whole-window 'looks flat' is expected even
when an anomaly exists → zoom in and check coupled channels", NOT "converge on flat").

EVIDENCE PRINCIPLES (how telemetry evidence actually works):
  • A value is anomalous only RELATIVE TO A BASELINE. Your primary lever is the CONTEXT you
    compare against — pick the one that makes the test DISCRIMINATING: the immediate past
    (shift vs drift), the SAME PHASE of a previous cycle (a real regime change vs normal
    periodic modulation), a matched NOMINAL stretch (is this interval truly off?), or the
    segment's own internal structure (the mechanism's signature).
  • A regime change is only visible by comparing BEFORE vs AFTER; a segment is not
    self-contained — its meaning depends on what you compare it to.
  • You have TWO directions, use BOTH: ZOOM IN (narrow the query onto a flagged
    sub-interval to localise a faint local event) and WIDEN OUT (enlarge the context to a
    broader span or a previous cycle to test whether a local blip is just normal structure).
  • CHARACTERISE EVERY NEW CONTEXT WITH BOTH BRANCHES. Every time you narrow or widen the
    range, you MUST read that new context with the ANALYST (get the numbers: change-points,
    d, KS, score) AND the PERCEPTOR (get the shape: modulation, form, cross-range likeness)
    BEFORE you refine a hypothesis or verify. Moving the range without re-interrogating both
    branches on it tells you nothing — that is the whole point of moving the range.
  • REFINE THE HYPOTHESIS ITSELF as evidence lands (re-`hypothesize`): sharpen a vague set
    into one specific mechanism, or broaden a confident claim when the wider context undercuts
    it. The hypothesis set should visibly change across steps, not stay fixed until verify.
  • Try to ANSWER THE QUESTION YOURSELF from the evidence so far; then query only what is
    still MISSING to separate the top hypotheses. A test every open hypothesis predicts is
    worthless.
"""

DECIDE_TASK = """\
=== TASK: CHOOSE THE NEXT MOVE ===
Pick exactly ONE action and give its coordinates:
  · describe    — ask the ANALYST to QUANTIFY something on query vs context (channels,
                  query_start/end, context_start/end, question). NON-BINDING. Use it for
                  TARGETED CONFIRMATION too, e.g. "is there a point in [q] decidedly OUT OF
                  LIMITS vs the context [c]?" (out_of_limits), or the size/significance of one
                  specific change — questions that get SHARPER as you close in.
  · perceive    — ask the PERCEPTOR what a segment LOOKS LIKE / compares to (channels, query
                  range, context_range = the range to compare against, question, probe_mode ∈
                  describe/compare/feature/cross_channel/localisation/discrimination).
                  Qualitative, NON-BINDING. Targeted too, e.g. "do you see a sudden spike / a
                  shape change here?".
  · hypothesize — (re)generate and REFINE your competing hypotheses. Do this FIRST (you start
                  with none), and again whenever new evidence sharpens or reframes them.
  · kg_lookup   — channels physically coupled to a channel (kg_channel).
  · conclude    — ACCEPT and stop: commit to your best-supported hypothesis (set hypothesis_id
                  to it; reasoning = the one-line justification). Do this ONLY when your
                  increasingly targeted questions have made the answer clear and the rivals are
                  ruled out. There is NO verifier — YOU judge when the evidence is enough.

SEPARATE WHY FROM WHAT: your `reasoning` states the OBJECTIVE of the move (why THIS query is the
next thing to learn, grounded in the KG) — it must NOT pre-describe what you expect to see; the
observation comes back from the agent. `reasoning` = ONE short sentence (≤25 words, single line);
it is logged and re-read every iteration. Output one action as JSON.
"""

# Two few-shot trajectories. THE POINT: every time the range is refined (narrowed OR widened),
# the orchestrator reads that NEW context with BOTH branches — analyst (numbers) AND perceptor
# (shape) — and only THEN refines the HYPOTHESIS (re-hypothesize). A goes coarse→fine (zoom in);
# B goes fine→coarse (widen out). Observations in (parentheses) are what the agents return.
FEWSHOT_TRAJECTORIES = """\
=== TWO WORKED TRAJECTORIES — the loop = {change range → READ IT WITH BOTH BRANCHES (describe +
perceive) → REFINE THE HYPOTHESIS} repeated, then verify. Never move the range without reading
what is there with both branches. You emit ONE action per turn. ===

Trajectory A — coarse→fine, ZOOM IN (two range refinements, each read by BOTH branches):
  seed: analyst "whole window flat, anomaly score ~0"; perceptor "smooth + slight modulation".
        KG: ~84% multivariate, anomalies mostly LOCAL.
  step1 decide=hypothesize — COARSE: {H1 coupling, H2 regime_change, H3 a local level/variance change}.
  --- range refinement #1: ZOOM to the candidate sub-interval [194,209] vs matched baseline [174,189]
  step2 decide=describe — "local events won't show whole-window; quantify this sub-interval vs its
        matched preceding baseline." → analyst per-channel Cohen's d + KS on [194,209] vs [174,189].
        (returns: only ch_B differs — NO mean offset, dominant metric trend/volatility; ch_A,ch_C flat)
  step3 decide=perceive — "now SEE the same new context: what is the shape doing there?" →
        probe_mode=compare on [194,209] vs [174,189]. (returns: on ch_B the periodic modulation
        attenuates; level unchanged; ch_A,ch_C look identical to baseline)
  step4 decide=hypothesize — REFINE #1 from BOTH reads: numbers say univariate-on-ch_B + no mean
        shift, shape says modulation attenuates → drop H1 coupling and level_shift; set SHARPENS to
        {H2 regime_change on ch_B, H3 variance_change/modulation-loss on ch_B}.
  --- range refinement #2: ZOOM tighter onto the transition itself, ch_B [198,206] vs [186,194]
  step5 decide=describe — "pin the mechanism: is the change abrupt or gradual, variance or shape?"
        → analyst on ch_B [198,206] vs [186,194]. (returns: variance ratio ↓, no trend, abrupt onset)
  step6 decide=perceive — "read that tighter context: sustained modulation loss or a one-off dropout?"
        → probe_mode=feature on ch_B [198,206]. (returns: modulation absent for the whole sub-interval)
  step7 decide=hypothesize — REFINE #2: abrupt sustained variance/modulation collapse at stable level
        → ONE specific mechanism: "variance_change (modulation loss) on ch_B, [198,206]".
  step8 decide=describe — TARGETED confirmation: "is any point in [198,206] decidedly out of limits
        vs [186,194], or is it a SUSTAINED variance drop?" → analyst out_of_limits + variance on ch_B
        [198,206] vs [186,194]. (returns: no single point out-of-limits; variance ratio ~0.3, the
        modulation band is collapsed across the whole sub-interval)
  step9 decide=conclude — ACCEPT: quant (sustained variance drop, no out-of-limits spike) and qual
        (modulation absent) agree and the rivals are out → conclude "variance_change / modulation-loss
        on ch_B, [198,206]". hypothesis_id=H3.

Trajectory B — fine→coarse, WIDEN OUT (two range refinements, each read by BOTH branches):
  seed: analyst flags a sharp peak at [250,262] on ch_X, large vs immediate neighbours.
  step1 decide=hypothesize — NARROW/confident: {H1 level_shift — a step/point anomaly at [250,262]}.
  --- range refinement #1: WIDEN to the same phase one cycle earlier, [250,262] vs [130,142]
  step2 decide=describe — "strong periodic modulation makes a peak-vs-neighbours saturate; compare to
        the same phase one cycle back." → analyst [250,262] vs [130,142]. (returns: Cohen's d small)
  step3 decide=perceive — "SEE that comparison: same recurring peak or a distinct new shape?" →
        probe_mode=discrimination on [250,262] vs [130,142]. (returns: same recurring peak shape)
  step4 decide=hypothesize — REFRAME from BOTH reads: the anomaly claim is undercut → question shifts
        from "which anomaly" to "is it one at all". Set BROADENS to {H1 normal periodic modulation
        (no anomaly), H2 a genuine amplitude increase vs baseline}.
  --- range refinement #2: WIDEN further to several cycles [0,300] to see the full periodic structure
  step5 decide=describe — "does the whole span behave periodically with [250,262] as one of many peaks?"
        → analyst segment/periodicity on [0,300]. (returns: regular period, [250,262] is a normal peak)
  step6 decide=perceive — "read the wide context: is [250,262] indistinguishable from the other peaks?"
        → probe_mode=describe on [0,300]. (returns: one of several identical periodic peaks)
  step7 decide=hypothesize — REFINE: converge to {H1 normal periodic modulation} as the leading
        explanation; H2 amplitude-increase is the only rival left to kill.
  step8 decide=describe — TARGETED confirmation: "is the [250,262] peak out of limits vs the
        matched-phase baseline [130,142]?" → analyst out_of_limits on ch_X [250,262] vs [130,142].
        (returns: peak within normal limits, |z|<2 vs the prior cycle — NOT out of limits)
  step9 decide=conclude — ACCEPT: no out-of-limits point and the shape matches the recurring peak
        → conclude "normal periodic modulation, no anomaly". hypothesis_id=H1.
"""

HYPOTHESIZE_TASK = """\
=== TASK: FORMULATE COMPETING HYPOTHESES ===
Produce 2–4 candidate explanations from the evidence gathered. Requirements: MUTUALLY
EXCLUSIVE; each with predicted_signature (what CONFIRMS it) and refuting_evidence (what
KILLS it); GROUNDED IN A MECHANISM; NOVEL patterns allowed (type "novel"); plausibilities
sum to 1.0. `type` ∈ the vocabulary, matching the mechanism (drift→gradual trend;
level_shift→a step; sensor_fault→a stuck actuator whose modulation collapses;
regime_change→distributional/shape change; variance_change→more variability;
cross_channel_coupling→two channels move together / lead-lag).
"""

class HypothesisDraft(BaseModel):
    id: str
    type: HypothesisType
    description: str
    predicted_signature: str
    refuting_evidence: str
    plausibility: float


class HypothesisSet(BaseModel):
    hypotheses: list[HypothesisDraft]


class Orchestrator(BaseLLMAgent):
    """The reasoning scientist. decide / hypothesize / update."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        kg: KnowledgeGraph | None = None,
        *,
        fallbacks: Sequence[BaseChatModel] | None = None,
    ) -> None:
        super().__init__(llm, fallbacks)
        self.kg = kg

    # ── decide the next move ─────────────────────────────────────────────────
    def decide(
        self,
        hypotheses: list[Hypothesis],
        evidence_log: list,
        channels: list[str],
        global_range: tuple[int, int],
    ) -> OrchestratorAction:
        human = (
            f"{self._kg_context(channels)}"
            f"Channels: {channels}. Global range: {global_range}.\n\n"
            f"Open hypotheses:\n{self._format_hypotheses(hypotheses, only_open=True)}\n\n"
            f"Investigation so far (analyst + perceptor + verdicts):\n{self._format_evidence(evidence_log)}\n\n"
            f"Reason FROM the mission knowledge above, then choose your next move."
        )
        system = ORCHESTRATOR_PREAMBLE + DECIDE_TASK + FEWSHOT_TRAJECTORIES
        action = self.structured(OrchestratorAction, system, human)
        assert isinstance(action, OrchestratorAction)
        # Guard: with no open hypotheses the only sensible move is to hypothesize.
        if not any(h.status == "open" for h in hypotheses) and action.action != "hypothesize":
            logger.info("decide: no open hypotheses → forcing hypothesize")
            action = OrchestratorAction(action="hypothesize", reasoning="no hypotheses yet")
        logger.info("decide: %s — %s", action.action, action.reasoning[:100])
        return action

    # ── hypothesize ──────────────────────────────────────────────────────────
    def hypothesize(
        self, evidence_log: list, eliminated: list[Hypothesis], channels: list[str]
    ) -> list[Hypothesis]:
        vocab = self.kg.hypothesis_vocabulary() if self.kg is not None else None
        vocab_line = f"Hypothesis vocabulary for this mission: {vocab}\n" if vocab else ""
        # Duck-typed: a rich KG (MissionKnowledgeGraph) supplies per-subsystem anomaly
        # priors that bias which mechanisms to expect — the KG informing hypothesize.
        prior_fn = getattr(self.kg, "hypothesis_prior", None)
        prior = prior_fn(channels) if callable(prior_fn) else ""
        human = (
            f"Evidence gathered so far:\n{self._format_evidence(evidence_log)}\n\n"
            f"Already-eliminated hypotheses:\n"
            f"{self._format_hypotheses(eliminated) if eliminated else '(none)'}\n\n"
            f"{prior}{vocab_line}Channels available: {channels}.\nFormulate the competing hypotheses now."
        )
        result = self.structured(HypothesisSet, ORCHESTRATOR_PREAMBLE + HYPOTHESIZE_TASK, human)
        assert isinstance(result, HypothesisSet)
        hyps = self._finalize_hypotheses(result.hypotheses)
        logger.info("hypothesize: %s", [(h.id, h.type, round(h.plausibility, 2)) for h in hyps])
        return hyps

    # ── formatting / helpers ─────────────────────────────────────────────────
    @staticmethod
    def _format_hypotheses(hyps: list[Hypothesis], only_open: bool = False) -> str:
        rows = [h for h in hyps if (h.status == "open" or not only_open)]
        if not rows:
            return "(none)"
        return "\n".join(
            f"- {h.id} [{h.type}] p={h.plausibility:.2f} status={h.status}: {h.description} "
            f"(confirm: {h.predicted_signature}; refute: {h.refuting_evidence})"
            for h in rows
        )

    # Only the last MAX_EVIDENCE_IN_PROMPT entries enter the prompt, each rationale
    # clipped — otherwise `decide` re-sends the whole growing log every iteration
    # (quadratic tokens → Groq's daily budget is exhausted mid-run).
    MAX_EVIDENCE_IN_PROMPT = 12
    RATIONALE_CLIP = 180

    @classmethod
    def _format_evidence(cls, log: list) -> str:
        if not log:
            return "(empty)"
        shown = log[-cls.MAX_EVIDENCE_IN_PROMPT :]
        elided = len(log) - len(shown)
        head = f"(… {elided} earlier entries elided …)\n" if elided else ""
        rows = "\n".join(
            f"- step {e.step} [{e.source}] H={e.hypothesis_id or '-'} {e.action} ch={e.channel} "
            f"q={e.query_range} c={e.context_range} -> {e.verdict}: {e.rationale[: cls.RATIONALE_CLIP]}"
            for e in shown
        )
        return head + rows

    @staticmethod
    def _finalize_hypotheses(drafts: list[HypothesisDraft]) -> list[Hypothesis]:
        total = sum(max(d.plausibility, 0.0) for d in drafts) or 1.0
        result: list[Hypothesis] = []
        seen: set[str] = set()
        for i, d in enumerate(drafts):
            hid = d.id or f"H{i + 1}"
            while hid in seen:
                hid += "'"
            seen.add(hid)
            result.append(
                Hypothesis(
                    id=hid, type=d.type, description=d.description,
                    predicted_signature=d.predicted_signature, refuting_evidence=d.refuting_evidence,
                    plausibility=max(d.plausibility, 0.0) / total, status="open",
                )
            )
        return result

    def _kg_context(self, channels: list[str]) -> str:
        """Assemble ALL available KG knowledge to prepend to the decide prompt: mission
        base-rates + per-subsystem anomaly priors + physically-coupled channels. The rich-KG
        methods are duck-typed, so a StubKnowledgeGraph (coupling only) still works."""
        if self.kg is None:
            return ""
        blocks: list[str] = []
        overview_fn = getattr(self.kg, "mission_overview", None)
        if callable(overview_fn):
            blocks.append(overview_fn())
        prior_fn = getattr(self.kg, "hypothesis_prior", None)
        if callable(prior_fn):
            blocks.append(prior_fn(channels))
        blocks.append(self._kg_channel_hint(channels))
        body = "".join(b for b in blocks if b)
        return f"=== MISSION KNOWLEDGE (consult before querying) ===\n{body}" if body else ""

    def _kg_channel_hint(self, channels: list[str]) -> str:
        assert self.kg is not None
        lines = [
            f"  {c} (subsystem {self.kg.subsystem_of(c)}): coupled with {rel}"
            for c in channels[:20]
            if (rel := self.kg.related_channels(c))
        ]
        if not lines:
            return ""
        return "Physically-coupled channels:\n" + "\n".join(lines) + "\n\n"
