"""Pure statistical tools — segmentation, per-interval metrics, cross-channel coupling.

PURE functions (§11.2): input arrays → output; no state, no LLM, no I/O. This is the
numeric engine the ANALYST's tools are built on (describe / anomaly_score / out_of_limits
/ cross_channel_coupling). PELT segmentation is computed ONCE per channel and cached. The
five per-metric names below name the signatures a hypothesis can present.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import ruptures as rpt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import ks_2samp, mannwhitneyu

from src import config


class Segment(TypedDict):
    """One homogeneous regime found by adaptive PELT."""

    start: int  # inclusive
    end: int  # exclusive
    n: int
    mean: float
    std: float
    slope: float  # robust per-sample slope (median of first differences)


# ─────────────────────────────────────────────────────────────────────────────
# Shared numeric helpers (pure)
# ─────────────────────────────────────────────────────────────────────────────
def as_array(series: object) -> np.ndarray:
    """Coerce to a 1-D float array."""
    return np.asarray(series, dtype=float).ravel()


def slice_range(series: np.ndarray, rng: tuple[int, int]) -> np.ndarray:
    """Slice [start, end) with clamping to the array bounds."""
    start, end = int(rng[0]), int(rng[1])
    start = max(0, start)
    end = min(series.size, end)
    if end <= start:
        return series[start:start]  # empty
    return series[start:end]


def saturate(value: float, scale: float) -> float:
    """Map a raw effect size to [0, 1] via min(1, |value| / scale)."""
    if scale <= 0:
        return 0.0
    return float(min(1.0, abs(value) / scale))


def robust_slope(x: np.ndarray) -> float:
    """Per-sample slope as the median of first differences.

    Robust to a single level step inside the window (the step is one outlier in
    the differences, ignored by the median) → ~0 for a step, ~m for a ramp.
    """
    if x.size < 2:
        return 0.0
    return float(np.median(np.diff(x)))


def local_sigma(x: np.ndarray) -> float:
    """Point-to-point noise scale, robust to a single step (MAD of first diffs).

    A step is one outlier in the differences, so this estimates the underlying
    noise, not the step height. Falls back to the plain std when degenerate.
    """
    if x.size < 2:
        return float(np.std(x)) if x.size else 0.0
    d = np.diff(x)
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma = 1.4826 * mad / math.sqrt(2.0)  # /sqrt(2): differencing doubles the variance
    if sigma <= 0.0:
        sigma = float(np.std(d) / math.sqrt(2.0))
    return float(max(sigma, 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Segment characterisation & adaptive PELT
# ─────────────────────────────────────────────────────────────────────────────
def characterize_segment(series: np.ndarray, start: int, end: int) -> Segment:
    """Summarise one segment: mean, std, robust slope, length."""
    x = series[start:end]
    n = int(x.size)
    if n == 0:
        return Segment(start=int(start), end=int(end), n=0, mean=0.0, std=0.0, slope=0.0)
    return Segment(
        start=int(start),
        end=int(end),
        n=n,
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        slope=robust_slope(x),
    )


def _within_regime_variance(x: np.ndarray, window: int) -> float:
    """Typical within-regime variance = median of sliding-window variances.

    Using the median makes it robust to the few windows that straddle a
    changepoint, and — crucially — it INCLUDES periodic modulation amplitude, so
    the segmenter does not split a modulated-but-stationary regime while still
    catching genuine level shifts (which raise the penalty's reference scale).
    """
    if x.size <= window:
        return float(np.var(x)) if x.size else 1.0
    windows = sliding_window_view(x, window)
    med_std = float(np.median(np.std(windows, axis=1)))
    return float(max(med_std, 1e-9) ** 2)


def _adaptive_penalty(x: np.ndarray) -> float:
    """BIC-like penalty for the l2 cost model: scale · within_regime_var · log(n).

    Scaling by the within-regime variance makes the segmenter adaptive across
    channels with different noise / modulation levels — one scale works everywhere.
    """
    n = x.size
    var = _within_regime_variance(x, config.PELT_MIN_SIZE)
    return float(config.PELT_PENALTY_SCALE * var * math.log(max(n, 2)))


def adaptive_pelt(series: object) -> list[Segment]:
    """Segment the series into homogeneous regimes with an adaptive penalty.

    Returns a list of Segments covering [0, n). A series too short to split
    returns a single segment.
    """
    x = as_array(series)
    n = x.size
    if n < 2 * config.PELT_MIN_SIZE:
        return [characterize_segment(x, 0, n)]

    pen = _adaptive_penalty(x)
    algo = rpt.Pelt(
        model=config.PELT_MODEL, min_size=config.PELT_MIN_SIZE, jump=config.PELT_JUMP
    ).fit(x)
    breakpoints: list[int] = algo.predict(pen=pen)  # ends of segments; last == n

    segments: list[Segment] = []
    prev = 0
    for bkp in breakpoints:
        segments.append(characterize_segment(x, prev, int(bkp)))
        prev = int(bkp)
    return segments


# ════════════════════════════════════════════════════════════════════════════
# Per-interval metrics, comparison & cross-channel coupling (was verification.py)
# ════════════════════════════════════════════════════════════════════════════

# Metric names — MUST match the SIGNATURE keys in src.adjudication.
MEAN_DEVIATION = "mean_deviation"
TREND_DIVERGENCE = "trend_divergence"
VOLATILITY_COLLAPSE = "volatility_collapse"
VOLATILITY_CHANGE = "volatility_change"
DISTRIBUTION_SHIFT = "distribution_shift"


class IntervalComparison(TypedDict):
    """Two-sample comparison of the query interval against the context."""

    n_query: int
    n_context: int
    n_effective: float
    mean_query: float
    mean_context: float
    std_query: float
    std_context: float
    cohens_d: float
    mann_whitney_u: float
    mann_whitney_p: float
    ks_stat: float
    ks_p: float


class AnomalyScore(TypedDict):
    """Composite score + per-metric breakdown (the analyst's anomaly_score tool)."""

    metrics: dict[str, float]
    dominant_metric: str
    composite: float
    power: float
    # supporting raw ingredients (for the rationale + audit)
    cohens_d: float
    slope_query: float
    slope_context: float
    std_query: float
    std_context: float
    effective_query_range: tuple[int, int]


class ContextualDeviation(TypedDict):
    """Level deviation of the query vs context, weighted by context stationarity."""

    deviation: float  # signed z-score of query level vs context
    context_n_segments: int  # scout changepoints strictly inside the context range
    context_stationary: bool
    weighted_deviation: float


class CrossChannelCoupling(TypedDict):
    """Lead/lag cross-correlation between two channels over the query window."""

    lag: int  # samples; +k means channel_b lags channel_a by k (a leads)
    peak_corr: float  # signed Pearson correlation at the peak-|corr| lag
    n_effective: float  # overlap length
    power: float


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _effective_sample_size(n_query: int, n_context: int) -> float:
    if n_query <= 0 or n_context <= 0:
        return 0.0
    return float(2.0 * n_query * n_context / (n_query + n_context))


def _cohens_d(q: np.ndarray, c: np.ndarray) -> float:
    if q.size < 2 or c.size < 2:
        return 0.0
    nq, nc = q.size, c.size
    vq, vc = float(np.var(q, ddof=1)), float(np.var(c, ddof=1))
    pooled = math.sqrt(max(((nq - 1) * vq + (nc - 1) * vc) / (nq + nc - 2), 0.0))
    if pooled <= 0.0:
        pooled = max(local_sigma(q), local_sigma(c), 1e-9)
    return float((np.mean(q) - np.mean(c)) / pooled)


def _detrended_std(x: np.ndarray) -> float:
    """Std of the residual after removing the robust linear trend.

    This is the quantity the volatility metrics use, so a drift (ramp) reads as
    trend_divergence — not volatility_change — while a modulation that stops reads
    as volatility_collapse (its residual variability drops).
    """
    if x.size < 2:
        return float(np.std(x)) if x.size else 0.0
    t = np.arange(x.size, dtype=float)
    resid = x - robust_slope(x) * t
    return float(np.std(resid))


def _ols_slope_se(x: np.ndarray) -> tuple[float, float]:
    """OLS slope of x vs its sample index and the slope's standard error.

    The t-statistic slope / SE is ~O(1) for pure noise and large for a real trend,
    so normalising by SE (not by window length) avoids amplifying slope noise.
    """
    n = x.size
    if n < 3:
        return 0.0, math.inf
    t = np.arange(n, dtype=float)
    t_c = t - t.mean()
    s_xx = float(np.dot(t_c, t_c))
    if s_xx <= 0.0:
        return 0.0, math.inf
    slope = float(np.dot(t_c, x - x.mean()) / s_xx)
    resid = x - (x.mean() + slope * t_c)
    resid_var = float(np.dot(resid, resid) / (n - 2))
    se = math.sqrt(resid_var / s_xx) if resid_var > 0 else 0.0
    return slope, se


def _trend_t(q: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Two-sample slope-difference t-statistic; also returns the two OLS slopes."""
    slope_q, se_q = _ols_slope_se(q)
    slope_c, se_c = _ols_slope_se(c)
    denom = math.sqrt(se_q**2 + se_c**2)
    t_stat = abs(slope_q - slope_c) / denom if denom > 0 else 0.0
    return t_stat, slope_q, slope_c


def _standardize(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    s = float(np.std(x))
    if s <= 0.0:
        return x - med
    return (x - med) / s


def _distribution_shift(q: np.ndarray, c: np.ndarray) -> float:
    """KS statistic on z-scored samples → pure shape difference in [0, 1]."""
    if q.size < 2 or c.size < 2:
        return 0.0
    stat, _ = ks_2samp(_standardize(q), _standardize(c))
    return float(stat)


def _representative_range(
    query_range: tuple[int, int], segments: list[Segment] | None
) -> tuple[int, int]:
    """Restrict a straddling query to its current (post-changepoint) regime.

    If scout segment boundaries fall strictly inside the query, keep only the
    portion after the last of them. Otherwise the query is returned unchanged.
    """
    if not segments:
        return query_range
    qs, qe = int(query_range[0]), int(query_range[1])
    internal = sorted({seg["start"] for seg in segments if qs < seg["start"] < qe})
    if not internal:
        return query_range
    return (internal[-1], qe)


# ─────────────────────────────────────────────────────────────────────────────
# tools
# ─────────────────────────────────────────────────────────────────────────────
def compare_intervals_statistics(
    series: object,
    query_range: tuple[int, int],
    context_range: tuple[int, int],
    segments: list[Segment] | None = None,
) -> IntervalComparison:
    """Mann-Whitney U, KS and Cohen's d between the query and context intervals."""
    x = as_array(series)
    q = slice_range(x, _representative_range(query_range, segments))
    c = slice_range(x, context_range)
    nq, nc = int(q.size), int(c.size)

    mean_q = float(np.mean(q)) if nq else 0.0
    mean_c = float(np.mean(c)) if nc else 0.0
    std_q = float(np.std(q)) if nq else 0.0
    std_c = float(np.std(c)) if nc else 0.0

    if nq >= 1 and nc >= 1:
        try:
            u_stat, u_p = mannwhitneyu(q, c, alternative="two-sided")
        except ValueError:  # all values identical → no rank information
            u_stat, u_p = 0.0, 1.0
        ks_stat, ks_p = ks_2samp(q, c)
    else:
        u_stat, u_p, ks_stat, ks_p = 0.0, 1.0, 0.0, 1.0

    return IntervalComparison(
        n_query=nq,
        n_context=nc,
        n_effective=_effective_sample_size(nq, nc),
        mean_query=mean_q,
        mean_context=mean_c,
        std_query=std_q,
        std_context=std_c,
        cohens_d=_cohens_d(q, c),
        mann_whitney_u=float(u_stat),
        mann_whitney_p=float(u_p),
        ks_stat=float(ks_stat),
        ks_p=float(ks_p),
    )


def _zero_score(effective_range: tuple[int, int]) -> AnomalyScore:
    metrics = {
        MEAN_DEVIATION: 0.0,
        TREND_DIVERGENCE: 0.0,
        VOLATILITY_COLLAPSE: 0.0,
        VOLATILITY_CHANGE: 0.0,
        DISTRIBUTION_SHIFT: 0.0,
    }
    return AnomalyScore(
        metrics=metrics,
        dominant_metric=MEAN_DEVIATION,
        composite=0.0,
        power=0.0,
        cohens_d=0.0,
        slope_query=0.0,
        slope_context=0.0,
        std_query=0.0,
        std_context=0.0,
        effective_query_range=effective_range,
    )


def _power(q: np.ndarray, c: np.ndarray) -> float:
    """power = sample_factor ∈ [0, 1]: enough effective samples to trust the test.

    Baseline reliability (a context spanning regimes) is a separate concern
    captured by compute_contextual_deviation and reported in the audit — it must
    not be folded in here, because a stationary *modulated* context is a perfectly
    good baseline yet has structured residuals that would fool a trend-based gate.
    """
    n_eff = _effective_sample_size(int(q.size), int(c.size))
    return float(min(1.0, n_eff / config.POWER_FULL_SAMPLES))


def compute_interval_anomaly_score(
    series: object,
    query_range: tuple[int, int],
    context_range: tuple[int, int],
    segments: list[Segment] | None = None,
) -> AnomalyScore:
    """Composite anomaly score + per-metric breakdown + dominant metric + power.

    composite == the dominant metric's score; dominant_metric == argmax over the
    five signatures (ties broken by insertion order for determinism).
    """
    x = as_array(series)
    effective_range = _representative_range(query_range, segments)
    q = slice_range(x, effective_range)
    c = slice_range(x, context_range)
    if q.size < 2 or c.size < 2:
        return _zero_score(effective_range)

    # Detrended (residual) std drives the volatility metrics so drift ≠ variance change.
    std_q = _detrended_std(q)
    std_c = _detrended_std(c)
    cohens_d = _cohens_d(q, c)

    # trend divergence: two-sample slope-difference t-statistic (SE-normalised).
    trend_t, slope_q, slope_c = _trend_t(q, c)

    # volatility: direction-split log-ratio of detrended stds.
    vol_log_sat = math.log(config.VOLATILITY_RATIO_SATURATION)
    log_ratio = math.log(max(std_q, 1e-12) / max(std_c, 1e-12))

    metrics = {
        MEAN_DEVIATION: saturate(cohens_d, config.COHEN_D_SATURATION),
        TREND_DIVERGENCE: saturate(trend_t, config.TREND_T_SATURATION),
        VOLATILITY_COLLAPSE: saturate(max(0.0, -log_ratio), vol_log_sat),
        VOLATILITY_CHANGE: saturate(max(0.0, log_ratio), vol_log_sat),
        DISTRIBUTION_SHIFT: _distribution_shift(q, c),
    }
    dominant = max(metrics, key=lambda k: metrics[k])

    return AnomalyScore(
        metrics=metrics,
        dominant_metric=dominant,
        composite=metrics[dominant],
        power=_power(q, c),
        cohens_d=cohens_d,
        slope_query=slope_q,
        slope_context=slope_c,
        std_query=std_q,
        std_context=std_c,
        effective_query_range=effective_range,
    )


def compute_contextual_deviation(
    series: object,
    query_range: tuple[int, int],
    context_range: tuple[int, int],
    segments: list[Segment] | None,
) -> ContextualDeviation:
    """Signed level deviation of the query vs context, weighted by how stationary
    the context is (a context spanning multiple scout regimes is a less reliable
    baseline, so its deviation is down-weighted)."""
    x = as_array(series)
    q = slice_range(x, _representative_range(query_range, segments))
    c = slice_range(x, context_range)
    if q.size < 1 or c.size < 1:
        return ContextualDeviation(
            deviation=0.0, context_n_segments=0, context_stationary=True, weighted_deviation=0.0
        )

    sig_c = local_sigma(c)
    deviation = float((np.median(q) - np.median(c)) / sig_c) if sig_c > 0 else 0.0

    cs, ce = int(context_range[0]), int(context_range[1])
    n_internal = (
        len({seg["start"] for seg in segments if cs < seg["start"] < ce}) if segments else 0
    )
    stationary = n_internal == 0
    weight = 1.0 if stationary else config.STATIONARITY_PENALTY

    return ContextualDeviation(
        deviation=deviation,
        context_n_segments=n_internal,
        context_stationary=stationary,
        weighted_deviation=deviation * weight,
    )


def compute_cross_channel_coupling(
    series_a: object,
    series_b: object,
    query_range: tuple[int, int],
    max_lag: int | None = None,
) -> CrossChannelCoupling:
    """Lead/lag coupling between two channels over the query window.

    Searches lags in [-max_lag, +max_lag] for the peak absolute Pearson correlation
    of channel_a against a shifted channel_b. A strong peak means the channels move
    together (the sign of `lag` says which leads) — the multivariate signature no
    univariate test can close.
    """
    lag_limit = config.CROSS_CORR_MAX_LAG if max_lag is None else max_lag
    a_full = slice_range(as_array(series_a), query_range)
    b_full = slice_range(as_array(series_b), query_range)
    n = int(min(a_full.size, b_full.size))
    if n < 3:
        return CrossChannelCoupling(lag=0, peak_corr=0.0, n_effective=float(n), power=0.0)
    a = a_full[:n]
    b = b_full[:n]

    best_lag = 0
    best_corr = 0.0
    for k in range(-lag_limit, lag_limit + 1):
        av, bv = (a[: n - k], b[k:]) if k >= 0 else (a[-k:], b[: n + k])
        if av.size < config.MIN_SAMPLES or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
            continue
        corr = float(np.corrcoef(av, bv)[0, 1])
        if abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, k

    power = min(1.0, n / config.POWER_FULL_SAMPLES)
    return CrossChannelCoupling(
        lag=best_lag, peak_corr=best_corr, n_effective=float(n), power=power
    )


# ════════════════════════════════════════════════════════════════════════════
# Targeted point-anomaly check — the analyst's `out_of_limits` tool. Non-binding:
# it QUANTIFIES how far the most-extreme query point sits from a context envelope,
# so the orchestrator can ask "is there a point decidedly out of limits here?".
# ════════════════════════════════════════════════════════════════════════════
class OutOfLimits(TypedDict):
    n_query: int
    max_abs_z: float  # largest |value - ctx_mean| / ctx_std over the query interval
    extreme_index: int  # absolute index of that most-extreme query point
    extreme_value: float
    n_out: int  # how many query points fall beyond ctx_mean ± limit_sigma·ctx_std
    limit_sigma: float
    out_of_limits: bool  # any query point beyond the limits


def point_out_of_limits(
    series: object,
    query_range: tuple[int, int],
    context_range: tuple[int, int],
    limit_sigma: float = config.OUT_OF_LIMITS_SIGMA,
) -> OutOfLimits:
    """Does any point in `query_range` sit OUT OF LIMITS vs `context_range`
    (context_mean ± limit_sigma · context_std)? A targeted, non-binding check."""
    x = as_array(series)
    q = x[query_range[0] : query_range[1]]
    c = x[context_range[0] : context_range[1]]
    ctx_mean = float(c.mean()) if c.size else 0.0
    ctx_std = float(c.std(ddof=1)) if c.size > 1 else 0.0
    if q.size == 0 or ctx_std == 0.0:
        return OutOfLimits(
            n_query=int(q.size), max_abs_z=0.0, extreme_index=query_range[0],
            extreme_value=float(q[0]) if q.size else 0.0, n_out=0, limit_sigma=limit_sigma,
            out_of_limits=False,
        )
    abs_z = np.abs((q - ctx_mean) / ctx_std)
    imax = int(abs_z.argmax())
    n_out = int((abs_z > limit_sigma).sum())
    return OutOfLimits(
        n_query=int(q.size), max_abs_z=float(abs_z[imax]), extreme_index=query_range[0] + imax,
        extreme_value=float(q[imax]), n_out=n_out, limit_sigma=limit_sigma, out_of_limits=n_out > 0,
    )
