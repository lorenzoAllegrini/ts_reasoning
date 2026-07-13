"""Central configuration — ALL constants live here (§11.2 principle 5).

No magic numbers anywhere else in the codebase. Thresholds, limits and weights
are defined once, here, and imported where needed. Changing the loop's behaviour
must be possible from this single file.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# LLM (§0.3) — the ONLY place these values are defined; read by src/llm.py.
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL: str = "openai/gpt-oss-20b"  # verified by src/smoke_test.py (§0.3 note)
LLM_TEMPERATURE: float = 0.0  # 0 everywhere: the loop must be reproducible
LLM_MAX_TOKENS: int = 4096  # NOT 512 (§8 bug #3)
STRUCTURED_OUTPUT_MAX_RETRIES: int = 3  # retry-with-validation on structured output (§0.4)
# gpt-oss-20b on Groq lowercases the function-calling tool name and Groq rejects it;
# json_schema (response_format) is schema-enforced and avoids that path entirely.
STRUCTURED_OUTPUT_METHOD: str = "json_schema"

# ─────────────────────────────────────────────────────────────────────────────
# Loop control / routing (§5.4) — deterministic, never decided by the LLM.
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEPS: int = 8  # verify rounds — the convergence budget
MAX_ORCHESTRATOR_ITERATIONS: int = 24  # hard cap on TOTAL agent moves (describe/perceive/
#                                        verify/…) → forces a report so the loop always ends
CONVERGENCE_THRESHOLD: float = 0.80
CONVERGENCE_MARGIN: float = 0.40
ELIMINATION_THRESHOLD: float = 0.05
ANALYST_MAX_TOOLS: int = 6  # cap on statistical tools the analyst runs per question
# LangGraph superstep budget: many supersteps per agent iteration — generous headroom.
RECURSION_LIMIT: int = 200

# ─────────────────────────────────────────────────────────────────────────────
# ChatTS perceptor (native TS-LLM). Loaded lazily on the first probe (~16 GB, MPS).
# ─────────────────────────────────────────────────────────────────────────────
CHATTS_CKPT: str = "ChatTS/ckpt"
CHATTS_MAX_POINTS: int = 512  # subsample the query slice to at most this many points
CHATTS_MAX_NEW_TOKENS: int = 160

# ─────────────────────────────────────────────────────────────────────────────
# Adjudication (§7.6) — the deterministic verdict rule. INV-1 lives here.
# ─────────────────────────────────────────────────────────────────────────────
MIN_POWER: float = 0.30  # below this the test lacks power → INCONCLUSIVE
MIN_SAMPLES: int = 20  # effective sample size below this → INCONCLUSIVE
SUPPORT_COMPOSITE_THRESHOLD: float = 0.70  # composite ≥ this + right signature → SUPPORTS
CONTRADICT_ABSENT_THRESHOLD: float = 0.30  # composite < this → predicted effect absent → CONTRADICTS

# ─────────────────────────────────────────────────────────────────────────────
# Belief reallocation (src/beliefs.py, used by the update node) — deterministic.
# Given the BINDING verdict, the active hypothesis's plausibility is multiplied by
# a factor, survivors are renormalised, and anything whose pre-renormalisation
# value falls below ELIMINATION_THRESHOLD is eliminated. The LLM only writes prose.
# ─────────────────────────────────────────────────────────────────────────────
SUPPORT_FACTOR: float = 3.0  # SUPPORTS(H) → plausibility(H) ×= this (then renormalise)
CONTRADICT_FACTOR: float = 0.05  # CONTRADICTS(H) → plausibility(H) ×= this (then renormalise)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical tools (tools/statistical.py, tools/verification.py) — pure functions.
# ─────────────────────────────────────────────────────────────────────────────
# PELT adaptive segmentation (ruptures). l2 = piecewise-constant mean; the adaptive
# penalty (∝ estimated noise variance · log n) makes it robust across channels with
# different noise levels. Volatility/shape changes are handled by the metrics below.
PELT_MODEL: str = "l2"
PELT_MIN_SIZE: int = 50  # minimum segment length (samples); ≥ modulation period so a
#                          modulated-but-stationary regime is not split into a staircase
PELT_JUMP: int = 5  # grid subsampling for speed
# Adaptive penalty: pen = PELT_PENALTY_SCALE * within_regime_var * log(n).
# Higher scale → fewer changepoints. Tuned so genuine level shifts are found but
# periodic modulation and noise are not.
PELT_PENALTY_SCALE: float = 5.0

# Per-metric [0,1] saturating maps. Each raw effect size divided by its saturation
# constant and clipped to [0,1]. dominant_metric = argmax; composite = that max.
COHEN_D_SATURATION: float = 2.0  # |Cohen's d| this large → mean_deviation score 1.0
TREND_T_SATURATION: float = 4.0  # a slope-difference t-statistic this large → trend score 1.0
#                                  (properly normalised by the slope SE, so noise ≈ 1, not huge)
VOLATILITY_RATIO_SATURATION: float = 2.0  # a ×2 std ratio (either dir) → volatility score 1.0

# power = sample_factor · context_stability.
POWER_FULL_SAMPLES: int = 40  # n_effective at/above which sample_factor saturates to 1.0
STATIONARITY_PENALTY: float = 0.5  # weight on deviation when the context spans regimes

# p-value below which a distributional test (KS / Mann-Whitney) is "significant".
SIGNIFICANCE_ALPHA: float = 0.05

# Cross-channel coupling (lead/lag) tool — multivariate signature.
CROSS_CORR_MAX_LAG: int = 60  # samples; search lags in [-max, +max] for the peak
COUPLING_SUPPORT_THRESHOLD: float = 0.70  # |peak corr| ≥ this → channels are coupled → SUPPORTS
COUPLING_ABSENT_THRESHOLD: float = 0.30  # |peak corr| < this → no coupling → CONTRADICTS

# Targeted point-anomaly check (analyst `out_of_limits` tool): a query point is OUT OF
# LIMITS if it lies beyond context_mean ± OUT_OF_LIMITS_SIGMA · context_std.
OUT_OF_LIMITS_SIGMA: float = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge graph (src/kg.py) — STUB reads channels.csv metadata ONLY (§ anti-leakage).
# Never read labels.csv or anomaly_types.csv: those are ground truth, not inputs.
# ─────────────────────────────────────────────────────────────────────────────
CHANNELS_CSV = "space-ai/datasets/ESA-Mission1/ESA-Mission1/channels.csv"
KG_MAX_RELATED_CHANNELS: int = 8  # cap the coupled set the plan passes together
