"""Module for optimizing loading matrix of the observation model.

This module provides functions for optimizing the loading matrix of the observation model,
including computing coherence, and scaling loading matrices to achieve target signal-to-noise ratios.
"""

import numpy as np
from typing import Optional, Callable, Tuple

from neurofisherSNR.utils import bias_matching_firing_rate, safe_normalize
from neurofisherSNR.snr import SNR_bound_instantaneous


def compute_coherence(CT: np.ndarray) -> float:
    """Compute coherence of loading matrix.

    Coherence is defined as the maximum absolute value of the off-diagonal entries of the normalized correlation matrix
    of the columns of C. Mathematically, for a matrix C with columns c_i, the coherence μ is:
        μ(C) = max_{i ≠ j} |<c_i, c_j>| / (||c_i|| * ||c_j||)
    where <c_i, c_j> is the inner product between columns i and j, and ||c_i|| is the norm of column i.
    See: https://en.wikipedia.org/wiki/Coherence_(signal_processing)

    Parameters
    ----------
    CT : ndarray
        Loading matrix

    Returns
    -------
    float
        Maximum off-diagonal correlation
    """
    assert isinstance(CT, np.ndarray) and CT.ndim == 2, "CT must be 2D ndarray"
    CT_norm = safe_normalize(CT)
    CC = CT_norm.T @ CT_norm
    return np.max(np.abs(CC - np.diag(np.diag(CC))))


def project_l1ball(v: np.ndarray, s: float = 1) -> np.ndarray:
    """Project vector onto L1-ball.

    Solves: min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v : ndarray
        Vector to project
    s : float, optional
        Ball radius, by default 1

    Returns
    -------
    ndarray
        Projected vector
    """
    assert isinstance(v, np.ndarray) and v.ndim == 1, "v must be 1D ndarray"
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    u = np.abs(v)
    if u.sum() <= s:
        return v
    w = project_simplex(u, s=s)
    w *= np.sign(v)
    return w


def project_simplex(v: np.ndarray, s: float = 1) -> np.ndarray:
    """Project vector onto positive simplex.

    Solves: min_w 0.5 * || w - v ||_2^2 , s.t. sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v : ndarray
        Vector to project
    s : float, optional
        Simplex radius, by default 1

    Returns
    -------
    ndarray
        Euclidean projection of v on the simplex

    References
    ----------
    [1] Duchi, John, et al. "Efficient projections onto the L1-ball for learning in high dimensions."
        Proceedings of the 25th international conference on Machine learning. 2008.
    """
    assert isinstance(v, np.ndarray) and v.ndim == 1, "v must be 1D ndarray"
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    if v.sum() == s and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w


def adjust_gain(
    x: np.ndarray,
    CT: np.ndarray,
    b: np.ndarray,
    current_gain: float,
    tgt_rate_per_bin: float,
    max_rate_per_bin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Limit firing rate to max_rate_per_bin.

    Parameters
    ----------
    x : ndarray
        Latent trajectory
    CT : ndarray
        Loading matrix
    b : ndarray
        Bias vector (1, num_neurons)
    tgt_rate_per_bin : float
        Target firing rate per bin
    max_rate_per_bin : float
        Maximum firing rate per bin

    Returns
    -------
    ndarray
        Mask of neurons to limit
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2, "x must be 2D ndarray"
    assert isinstance(CT, np.ndarray) and CT.ndim == 2, "CT must be 2D ndarray"
    if isinstance(b, np.ndarray) and b.ndim == 1:
        b = b.reshape(1, -1)
    assert (
        isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[0] == 1
    ), "b must be 2D ndarray (1, num_neurons)"
    assert isinstance(current_gain, float) or isinstance(
        current_gain, int
    ), "current_gain must be float or int"
    assert tgt_rate_per_bin > 0.0, "tgt_rate_per_bin must be positive"
    assert max_rate_per_bin > 0.0, "max_rate_per_bin must be positive"
    Cx = (x @ CT).max(axis=0)
    CT = CT * current_gain
    b, firing_rates = bias_matching_firing_rate(x, CT, b, tgt_rate=tgt_rate_per_bin)
    adjusted_idx = firing_rates.max(axis=0) > max_rate_per_bin
    adjusted_gain = np.ones(CT.shape[1]) * current_gain
    adjusted_gain[adjusted_idx] = (
        current_gain
        + np.log(max_rate_per_bin / firing_rates.max(axis=0)[adjusted_idx])
        / Cx[adjusted_idx]
    )
    return adjusted_gain, adjusted_idx


def _logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return np.log(np.mean(np.exp(values - max_value), axis=axis, keepdims=True)) + max_value


def _bias_match_from_drive(
    log_drive: np.ndarray,
    tgt_rate_per_bin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match per-neuron mean firing rates for a precomputed log drive.

    Parameters
    ----------
    log_drive : ndarray
        Matrix ``x @ CT`` with shape ``(n_timepoints, d_neurons)``.
    tgt_rate_per_bin : float
        Target mean rate per time bin.

    Returns
    -------
    tuple
        ``(b, rates)`` with ``b`` shape ``(1, d_neurons)`` and ``rates`` shape
        ``(n_timepoints, d_neurons)``.
    """
    b = np.log(float(tgt_rate_per_bin)) - _logmeanexp(log_drive, axis=0)
    return b, np.exp(log_drive + b)


def _row_gain_caps(
    x: np.ndarray,
    CT: np.ndarray,
    tgt_rate_per_bin: float,
    max_rate_per_bin: float,
    max_gain: float,
) -> np.ndarray:
    """Return per-neuron gain caps that satisfy the max-rate constraint.

    The cap is computed after bias matching, because changing the loading gain
    changes both the rate variance and the bias required to preserve the mean.
    """
    log_max_rate = float(np.log(max_rate_per_bin))
    log_drive = x @ CT

    def max_log_rate_for(row_gains: np.ndarray) -> np.ndarray:
        scaled_drive = log_drive * row_gains.reshape(1, -1)
        b = np.log(float(tgt_rate_per_bin)) - _logmeanexp(scaled_drive, axis=0)
        return np.max(scaled_drive + b, axis=0)

    low = np.zeros(CT.shape[1], dtype=np.float64)
    high = np.ones(CT.shape[1], dtype=np.float64)
    max_gain = float(max_gain)

    for _ in range(80):
        max_log_rate = max_log_rate_for(high)
        feasible = max_log_rate <= log_max_rate + 1e-10
        expandable = feasible & (high < max_gain)
        if not bool(np.any(expandable)):
            break
        low[expandable] = high[expandable]
        high[expandable] = np.minimum(2.0 * high[expandable], max_gain)

    max_log_rate = max_log_rate_for(high)
    bounded = max_log_rate > log_max_rate + 1e-10
    if not bool(np.any(bounded)):
        return np.full(CT.shape[1], max_gain, dtype=np.float64)

    for _ in range(50):
        mid = 0.5 * (low + high)
        max_log_rate = max_log_rate_for(mid)
        too_high = bounded & (max_log_rate > log_max_rate)
        high[too_high] = mid[too_high]
        low[bounded & ~too_high] = mid[bounded & ~too_high]

    caps = np.full(CT.shape[1], max_gain, dtype=np.float64)
    caps[bounded] = low[bounded]
    return caps


def optimize_C(
    x: np.ndarray,
    C: np.ndarray,
    b: np.ndarray,
    tgt_rate_per_bin: float,
    max_rate_per_bin: float,
    tgt_snr: float,
    snr_fn: Callable = SNR_bound_instantaneous,
    priority: str = "max",
    max_iter: int = 40,
    tol: float = 0.1,
    min_gain: float = 0.5,
    max_gain: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Scale a loading matrix to match a target SNR.

    The optimizer searches a scalar gain. With ``priority="max"``, each neuron
    is capped at the largest row gain that satisfies ``max_rate_per_bin`` after
    bias matching. Every candidate SNR is computed from the exact returned
    loading and bias, so ``(C, b, achieved_snr)`` is internally consistent.

    Args:
        x: The latent trajectory matrix, shape ``(n_timepoints, d_latent)``.
        C: Loading matrix, shape ``(d_neurons, d_latent)``.
        b: Bias vector, shape ``(1, d_neurons)``. Used for shape validation;
            the returned bias is recomputed by mean-rate matching.
        tgt_rate_per_bin: Target mean firing rate per bin.
        max_rate_per_bin: Maximum firing rate per bin.
        tgt_snr: Target signal-to-noise ratio in dB.
        snr_fn: Function to compute SNR from ``(x, C.T, b)``.
        priority: ``"max"`` enforces the max-rate cap; ``"mean"`` performs a
            pure scalar-gain search with mean-rate bias matching.
        max_iter: Maximum number of bisection iterations.
        tol: Relative tolerance for SNR matching.
        min_gain: Initial minimum scalar gain for search.
        max_gain: Initial maximum scalar gain for search.
        verbose: Whether to print debug information.

    Returns:
        Tuple of ``(scaled_C, updated_b, achieved_snr)``.

    Raises:
        ValueError: If arguments are invalid or no finite candidate is found.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2, "x must be 2D ndarray"
    assert isinstance(C, np.ndarray) and C.ndim == 2, "C must be 2D ndarray"
    if isinstance(b, np.ndarray) and b.ndim == 1:
        b = b.reshape(1, -1)
        print("WARNING: b is reshaped to (1, num_neurons)")
    assert (
        isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[0] == 1
    ), "b must be 2D ndarray (1, num_neurons)"
    assert x.shape[1] == C.shape[1], "x and C must have matching latent dimensions"
    assert b.shape[1] == C.shape[0], "b and C must have matching neuron dimensions"
    assert tgt_rate_per_bin > 0.0, "tgt_rate_per_bin must be positive"
    assert max_rate_per_bin > 0.0, "max_rate_per_bin must be positive"
    assert tgt_rate_per_bin <= max_rate_per_bin, (
        "tgt_rate_per_bin must be no larger than max_rate_per_bin"
    )
    assert isinstance(tgt_snr, float) or isinstance(
        tgt_snr, int
    ), "tgt_snr must be float or int"
    assert callable(snr_fn), "snr_fn must be callable"
    assert isinstance(priority, str), "priority must be a string"
    if priority not in {"mean", "max"}:
        raise ValueError(f"priority must be 'mean' or 'max', got {priority!r}")
    assert (
        isinstance(max_iter, int) and max_iter > 0
    ), "max_iter must be positive integer"
    assert 0.0 < tol < 1.0, "tol must be between 0 and 1"
    assert min_gain > 0.0 and max_gain > 0.0, "gains must be positive"

    CT = C.T.astype(np.float64, copy=True)
    initial_min_gain = float(min_gain)
    initial_max_gain = float(max_gain)
    max_search_gain = max(initial_min_gain, initial_max_gain, 1.0) * 2.0 ** max(
        int(max_iter), 10
    )
    row_caps = (
        _row_gain_caps(
            x,
            CT,
            float(tgt_rate_per_bin),
            float(max_rate_per_bin),
            max_search_gain,
        )
        if priority == "max"
        else np.full(CT.shape[1], max_search_gain, dtype=np.float64)
    )

    def evaluate(scalar_gain: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        row_gains = np.minimum(float(scalar_gain), row_caps)
        curr_CT = CT * row_gains.reshape(1, -1)
        curr_b, _rates = _bias_match_from_drive(x @ curr_CT, float(tgt_rate_per_bin))
        curr_snr = float(snr_fn(x, curr_CT, curr_b))
        return curr_CT, curr_b, curr_snr, row_gains

    def update_best(candidate, best):
        if not np.isfinite(candidate[2]):
            return best
        if best is None or abs(candidate[2] - tgt_snr) < abs(best[2] - tgt_snr):
            return candidate
        return best

    try:
        low_gain = initial_min_gain
        low = evaluate(low_gain)
        best = update_best(low, None)
        for _ in range(max_iter):
            if low[2] <= tgt_snr or low_gain <= 1e-12:
                break
            low_gain *= 0.5
            low = evaluate(low_gain)
            best = update_best(low, best)

        high_gain = initial_max_gain
        high = evaluate(high_gain)
        best = update_best(high, best)
        for _ in range(max_iter):
            if high[2] >= tgt_snr or high_gain >= max_search_gain:
                break
            high_gain = min(2.0 * high_gain, max_search_gain)
            high = evaluate(high_gain)
            best = update_best(high, best)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Failed to evaluate initial SNR search bounds") from exc

    if best is None:
        raise ValueError(f"Failed to find solution for target SNR {tgt_snr} dB")
    if low[2] >= tgt_snr or high[2] <= tgt_snr:
        if verbose:
            print(
                f"Target SNR {tgt_snr:.2f} dB is outside feasible range; "
                f"using SNR = {best[2]:.2f} dB"
            )
        return best[0].T, best[1], best[2]

    denom = max(abs(float(tgt_snr)), 1e-12)
    for i in range(max_iter):
        curr_gain = np.sqrt(low_gain * high_gain)
        try:
            current = evaluate(curr_gain)
        except np.linalg.LinAlgError:
            high_gain = curr_gain
            continue
        best = update_best(current, best)
        rel_err = abs(current[2] - tgt_snr) / denom
        if verbose:
            capped = current[3] < curr_gain
            print(
                f"SNR: {current[2]:.2f} dB, Gain: {curr_gain:.2f}, "
                f"Adjusted neurons: {int(capped.sum())}"
            )
        if rel_err <= tol:
            if verbose:
                print(
                    f"Converged after {i + 1} iterations with relative error {rel_err:.2%}"
                )
            return current[0].T, current[1], current[2]
        if current[2] < tgt_snr:
            low_gain = curr_gain
        else:
            high_gain = curr_gain
        if (high_gain - low_gain) / low_gain < 1e-6:
            break

    if verbose:
        print(
            f"Could not find solution for target SNR {tgt_snr} dB, "
            f"using best solution found: SNR = {best[2]:.2f} dB"
        )
    return best[0].T, best[1], best[2]


def initialize_C(
    d_latent: int,
    d_neurons: int,
    p_coh: float,
    p_sparse: float = 0.0,
) -> np.ndarray:
    """Generate loading matrix with controllable coherence and sparsity.

    Parameters
    ----------
    d_latent : int
        Latent dimension
    d_neurons : int
        Number of neurons
    p_coh : float
        Target coherence
    p_sparse : float, optional
        Sparsity probability, by default 0
    C : ndarray, optional
        Initial matrix, by default None

    Returns
    -------
    ndarray
        Generated loading matrix
    """
    assert (
        isinstance(d_latent, int) and d_latent > 0
    ), "d_latent must be positive integer"
    assert (
        isinstance(d_neurons, int) and d_neurons > 0
    ), "d_neurons must be positive integer"
    assert 0.0 <= p_sparse <= 1.0, "p_sparse must be between 0 and 1"
    assert isinstance(p_coh, float) or isinstance(
        p_coh, int
    ), "p_coh must be float or int"

    CT = np.random.randn(d_latent, d_neurons)
    CT = CT * (np.random.rand(d_latent, d_neurons) > p_sparse)
    CT = safe_normalize(CT)
    CT[np.isnan(CT)] = 0

    n_iter = 15
    n_inner = 1000
    rho = 0.5
    eta = 1.1
    lbda = 0.9
    alpha = lbda * rho

    for _ in range(n_iter):
        for _ in range(n_inner):
            coh = compute_coherence(CT)
            if coh < p_coh:
                CT = safe_normalize(CT)
                CT[np.isnan(CT)] = 0
                return CT.T

            vv = (CT.T @ CT - np.eye(d_neurons)) / rho
            v = project_l1ball(vv.flatten(), s=1)
            v_mat = np.reshape(v, (d_neurons, d_neurons))

            mm = CT - alpha * CT @ (v_mat + v_mat.T)
            CT = safe_normalize(mm)
            CT[np.isnan(CT)] = 0
        rho = rho / eta
        alpha = lbda * rho

    if coh >= p_coh:
        print(f"WARNING: target Coherence {p_coh} not reached, Current Coherence {coh}")

    CT = safe_normalize(CT)
    CT[np.isnan(CT)] = 0
    return CT.T
