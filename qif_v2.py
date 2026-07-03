import argparse
import os
import re

import numpy as np
import scipy.linalg
from scipy import signal
from scipy.interpolate import BSpline
from scipy.signal import medfilt
from scipy.optimize import minimize

ALPHA_LOG_MIN = -80.0
ALPHA_LOG_MAX = 20.0


# ---------------------------
# Utilities
# ---------------------------

def hermitize(A: np.ndarray) -> np.ndarray:
    """Return (A + A^H)/2 to enforce Hermiticity numerically."""
    return 0.5 * (A + A.conj().T)


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phases to (-pi, pi] for numerical stability of exp(1j*phi)."""
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------
# B-spline evaluation (axis=0 convention)
# ---------------------------

def evaluate_spline(coeffs: np.ndarray, knots: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Evaluate cubic B-spline with axis=0 spline dimension.

    coeffs: shape (N_coeff, ...)
    knots:  shape (N_coeff + 4,) for k=3
    freqs:  shape (N_f,)

    Returns: shape (N_f, ...)
    """
    spl = BSpline(knots, coeffs, k=3, axis=0, extrapolate=False)
    return spl(freqs)


# ---------------------------
# Bin integration: fixed 10-subinterval Simpson's rule (11 points)
# ---------------------------

_SIMPSON10_WEIGHTS = np.array([1, 4, 2, 4, 2, 4, 2, 4, 2, 4, 1], dtype=float)

def simpson10_bin_average(y: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    """
    Compute (1/Δf) ∫ y(f) df over each bin using fixed 10-subinterval composite Simpson's rule.

    y:     shape (N_f, 11) values at equally spaced points in each bin.
    f_min: shape (N_f,)
    f_max: shape (N_f,)

    Returns: shape (N_f,) bin-averaged integral.
    """
    df = f_max - f_min
    if np.any(df <= 0):
        raise ValueError("All bins must satisfy f_max > f_min.")
    h = df / 10.0  # step size per bin
    integral = (h / 3.0) * np.sum(y * _SIMPSON10_WEIGHTS[None, :], axis=1)
    return integral / df


def bin_grid_11(f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    """
    Build the 11-point (N_f, 11) grid for Simpson10 over each bin.
    """
    t = np.linspace(0.0, 1.0, 11, dtype=float)  # 11 points
    return f_min[:, None] + (f_max - f_min)[:, None] * t[None, :]


def compute_T_eff_sq_simpson10(T_sq_func, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    """
    Compute T_eff^2 per bin as (1/Δf) ∫ |T(f)|^2 df via Simpson10.

    T_sq_func: callable mapping ndarray freqs -> ndarray |T(f)|^2 (same shape)
    """
    f_grid = bin_grid_11(f_min, f_max)
    y = T_sq_func(f_grid)
    if y.shape != f_grid.shape:
        raise ValueError("T_sq_func must return an array of the same shape as its input.")
    return simpson10_bin_average(y, f_min, f_max)


def compute_T_over_f2_avg_simpson10(T_sq_func, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    """
    Compute (1/Δf) ∫ |T(f)|^2 / f^2 df via Simpson10.

    This is the preferred, bias-minimizing bin-averaged factor to multiply the f^-2 model.
    """
    f_grid = bin_grid_11(f_min, f_max)
    y = T_sq_func(f_grid) / np.maximum(f_grid, 1e-300) ** 2
    return simpson10_bin_average(y, f_min, f_max)


def compute_T_over_fgamma_avg_simpson10(
    T_sq_func,
    f_min: np.ndarray,
    f_max: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute (1/Δf) ∫ |T(f)|^2 / f^gamma df via Simpson10.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    f_grid = bin_grid_11(f_min, f_max)
    y = T_sq_func(f_grid) / np.maximum(f_grid, 1e-300) ** gamma
    return simpson10_bin_average(y, f_min, f_max)


# ---------------------------
# Fixed thresholds (computed once from on-source data; reused in bootstrap)
# ---------------------------

def compute_fixed_thresholds(
    S_hat: np.ndarray,
    valid: np.ndarray,
    rel_P_floor: float = 1e-3,
    rel_B_clip: float = 100.0,
    medfilt_kernel: int = 21,
    abs_floor: float = 1e-50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fixed, robust scale thresholds from on-source data only (valid bins).

    Returns:
        P_floor_val: shape (N_f, 3)
        B_clip_val:  shape (N_f,)
    """
    if S_hat.ndim != 3 or S_hat.shape[1:] != (3, 3):
        raise ValueError("S_hat must have shape (N_f, 3, 3).")
    Nf = S_hat.shape[0]
    if valid.shape != (Nf,):
        raise ValueError("valid mask must have shape (N_f,)")

    # Extract diagonals (real, non-negative)
    diag = np.real(np.stack([np.diag(S_hat[k]) for k in range(Nf)], axis=0))
    diag = np.maximum(diag, 0.0)

    # Fill invalid bins with per-channel median of valid bins (robust to NaN poisoning)
    diag_filled = diag.copy()
    for i in range(3):
        med_i = np.median(diag[valid, i]) if np.any(valid) else 0.0
        diag_filled[~valid, i] = med_i

    # Median filter smoothing per channel (kernel size must be odd)
    if medfilt_kernel % 2 == 0:
        medfilt_kernel += 1
    baseline = np.zeros_like(diag_filled)
    for i in range(3):
        baseline[:, i] = medfilt(diag_filled[:, i], kernel_size=medfilt_kernel)

    baseline = np.maximum(baseline, abs_floor)

    # Relative floor on instrument diagonal
    P_floor_val = np.maximum(abs_floor, rel_P_floor * baseline)

    # Clip threshold for env (diag(BB^H) units): based on max baseline diagonal across channels
    baseline_max = np.max(baseline, axis=1)
    B_clip_val = np.maximum(abs_floor, rel_B_clip * baseline_max)

    return P_floor_val, B_clip_val


# ---------------------------
# QIF spectral model (bin-averaged)
# ---------------------------

def qif_path_psd_binavg(
    alpha: float,
    c0: float,
    lP: float,
    L: float,
    f_min: np.ndarray,
    f_max: np.ndarray,
    T_eff_sq: np.ndarray | None = None,
    T_over_f2_avg: np.ndarray | None = None,
    T_over_fgamma_avg: np.ndarray | None = None,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
) -> np.ndarray:
    """
    Bin-averaged one-sided strain PSD per *path* under a power-law model:
        S_h(f) = A_h * (|T(f)|^2 / f^gamma).

    By default, alpha is interpreted as A_h (units: strain^2/Hz * Hz^2).
    If use_planck_scale=True, alpha is treated as a dimensionless scale and
    mapped via:
        A_h = (alpha * c0 * lP) / (2*pi^2*L^2).

    Provide either:
      - T_over_f2_avg[k] = (1/Δf)∫ |T(f)|^2/f^2 df  (preferred), or
      - T_eff_sq[k]      = (1/Δf)∫ |T(f)|^2 df      (then use analytic avg 1/(f_min*f_max)).

    Returns: S_path[k] (strain^2/Hz), shape (N_f,)
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if use_planck_scale:
        A = (alpha * c0 * lP) / (2.0 * np.pi**2 * L**2)
    else:
        A = alpha

    if T_over_fgamma_avg is not None:
        return A * T_over_fgamma_avg
    if T_over_f2_avg is not None and np.isclose(gamma, 2.0):
        return A * T_over_f2_avg

    if T_eff_sq is None:
        # If no transfer info is provided, user must have pre-masked band where T≈1.
        T_eff_sq = np.ones_like(f_min, dtype=float)

    # If only T_eff_sq is available, assume |T|^2 is approximately constant over the bin.
    if np.isclose(gamma, 1.0):
        denom = np.maximum(np.log(np.maximum(f_max, 1e-300)) - np.log(np.maximum(f_min, 1e-300)), 1e-300)
        avg = denom / np.maximum(f_max - f_min, 1e-300)
    else:
        denom = np.maximum(1.0 - gamma, 1e-9)
        avg = (np.maximum(f_max, 1e-300) ** (1.0 - gamma) - np.maximum(f_min, 1e-300) ** (1.0 - gamma)) / denom
        avg = avg / np.maximum(f_max - f_min, 1e-300)
    return A * T_eff_sq * avg


# ---------------------------
# Likelihood (complex Wishart approx)
# ---------------------------

def loglike_et_qif(
    # global amplitude: alpha_val is log(alpha) or -inf to force alpha=0 (env-only)
    alpha_val: float,
    # spline coefficients
    rho_coeffs: np.ndarray,        # (N_coeff, 1)
    P_coeffs: np.ndarray,          # (N_coeff, 3)
    B_real_coeffs: np.ndarray,     # (N_coeff, 3, r)
    B_imag_coeffs: np.ndarray,     # (N_coeff, 3, r)
    phi_coeffs: np.ndarray,        # (N_coeff, 2)
    # data and frequency grid
    S_hat: np.ndarray,             # (N_f, 3, 3)
    m_eff: np.ndarray,             # (N_f,)
    freqs: np.ndarray,             # (N_f,)
    f_min: np.ndarray,             # (N_f,)
    f_max: np.ndarray,             # (N_f,)
    knots: np.ndarray,
    # constants
    L: float,
    c0: float,
    lP: float,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    # transfer integration (choose one)
    T_eff_sq: np.ndarray | None = None,            # (N_f,)
    T_over_f2_avg: np.ndarray | None = None,       # (N_f,) gamma=2 only
    T_over_fgamma_avg: np.ndarray | None = None,   # (N_f,) general gamma
    # fixed thresholds (must be computed once from on-source and held fixed)
    P_floor: np.ndarray | None = None,          # (N_f, 3)
    B_clip: np.ndarray | None = None,           # (N_f,)
    # numerics
    eps: float = 1e-9,
    jitter_floor: float = 1e-50,
    # Overflow guards only: exp() overflows near 709. These clips must NOT act as
    # physical priors -- strain PSDs live at log P ~ -108, so tight clips silently
    # clamp the model far above the data scale.
    clip_logP: float = 700.0,
    clip_logit_rho: float = 50.0,
    clip_log_alpha: float = 700.0,
    B_elem_clip_factor: float = 1e6,
) -> float:
    """
    Commissioning-ready ET QIF log-likelihood (complex Wishart approximation).
    Returns log-likelihood up to an additive constant.

    Notes:
      - S_hat is symmetrized per-bin.
      - Splines are evaluated with extrapolate=False; any NaNs -> reject.
      - Instrument diagonal is floored by P_floor (fixed from on-source).
      - Environmental factors are elementwise clipped (overflow guard) AND then rescaled to satisfy B_clip.
      - Calibration gains are phase-only, gauge-fixed (G1=1); phases are wrapped.
    """
    if S_hat.ndim != 3 or S_hat.shape[1:] != (3, 3):
        raise ValueError("S_hat must have shape (N_f, 3, 3).")
    Nf = S_hat.shape[0]
    for arr, name in [(m_eff, "m_eff"), (freqs, "freqs"), (f_min, "f_min"), (f_max, "f_max")]:
        if arr.shape != (Nf,):
            raise ValueError(f"{name} must have shape (N_f,)")

    # --- splines ---
    P_log = evaluate_spline(P_coeffs, knots, freqs)
    if np.any(~np.isfinite(P_log)):
        return -np.inf
    P = np.exp(np.clip(P_log, -clip_logP, clip_logP))
    if P_floor is not None:
        if P_floor.shape != (Nf, 3):
            raise ValueError("P_floor must have shape (N_f, 3)")
        P = np.maximum(P, P_floor)
    else:
        P = np.maximum(P, jitter_floor)

    rho_logit = evaluate_spline(rho_coeffs, knots, freqs).reshape(-1)
    if np.any(~np.isfinite(rho_logit)):
        return -np.inf
    rho = 1.0 / (1.0 + np.exp(-np.clip(rho_logit, -clip_logit_rho, clip_logit_rho)))
    rho = np.clip(rho, 1e-6, 1.0 - 1e-6)

    B_real = evaluate_spline(B_real_coeffs, knots, freqs)
    B_imag = evaluate_spline(B_imag_coeffs, knots, freqs)
    if np.any(~np.isfinite(B_real)) or np.any(~np.isfinite(B_imag)):
        return -np.inf

    phi_23 = evaluate_spline(phi_coeffs, knots, freqs)
    if np.any(~np.isfinite(phi_23)):
        return -np.inf
    phi_23 = wrap_phase(phi_23)

    # --- alpha ---
    if np.isneginf(alpha_val):
        alpha = 0.0
    else:
        alpha = float(np.exp(np.clip(alpha_val, -clip_log_alpha, clip_log_alpha)))

    # --- QIF bin-avg per-path PSD ---
    if alpha > 0:
        S_path = qif_path_psd_binavg(
            alpha,
            c0,
            lP,
            L,
            f_min,
            f_max,
            T_eff_sq=T_eff_sq,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
        )
    else:
        S_path = np.zeros(Nf, dtype=float)

    # elementwise clip for env factors (overflow guard), then safe rescaling clip
    if B_clip is not None:
        if B_clip.shape != (Nf,):
            raise ValueError("B_clip must have shape (N_f,)")
        B_elem_clip = B_elem_clip_factor * np.sqrt(np.maximum(B_clip, jitter_floor))
        B_elem_clip = B_elem_clip.reshape(Nf, 1, 1)
        B_real = np.clip(B_real, -B_elem_clip, B_elem_clip)
        B_imag = np.clip(B_imag, -B_elem_clip, B_elem_clip)

    # ---- fully vectorized Wishart log-likelihood over bins ----
    eye3 = np.eye(3, dtype=complex)

    # environment (rank r) with per-bin rescaling clip
    B = (B_real + 1j * B_imag).astype(complex)              # (Nf, 3, r)
    if B_clip is not None:
        diag_B = np.sum(np.abs(B) ** 2, axis=2)             # (Nf, 3) = diag(BB^H)
        dmax = np.max(diag_B, axis=1)                       # (Nf,)
        limit = np.asarray(B_clip, dtype=float)
        scale = np.ones(Nf)
        need = np.isfinite(dmax) & np.isfinite(limit) & (dmax > limit) & (dmax > 0.0) & (limit > 0.0)
        scale[need] = np.sqrt(limit[need] / dmax[need])
        B = B * scale[:, None, None]
    Sigma_env = B @ np.conjugate(np.transpose(B, (0, 2, 1)))  # (Nf, 3, 3)

    # instrument
    Sigma = Sigma_env.copy()
    Sigma[:, 0, 0] += P[:, 0]
    Sigma[:, 1, 1] += P[:, 1]
    Sigma[:, 2, 2] += P[:, 2]

    # QIF template M(rho): diag 2, off-diag -rho
    if alpha > 0:
        J_off = np.ones((3, 3)) - np.eye(3)
        M = 2.0 * np.eye(3)[None, :, :] - rho[:, None, None] * J_off[None, :, :]
        Sigma = Sigma + (S_path[:, None, None] * M).astype(complex)

    # calibration (phase-only, gauge-fixed): Sigma -> G Sigma G^H elementwise
    g = np.stack([np.ones(Nf, dtype=complex),
                  np.exp(1j * phi_23[:, 0]),
                  np.exp(1j * phi_23[:, 1])], axis=1)        # (Nf, 3)
    Sigma = Sigma * (g[:, :, None] * np.conjugate(g[:, None, :]))

    # hermitize model and data
    Sigma = 0.5 * (Sigma + np.conjugate(np.transpose(Sigma, (0, 2, 1))))
    S_sym = 0.5 * (S_hat + np.conjugate(np.transpose(S_hat, (0, 2, 1))))

    # jitter hardening
    max_diag = np.max(np.real(np.diagonal(Sigma, axis1=1, axis2=2)), axis=1)
    Sigma = Sigma + (eps * max_diag + jitter_floor)[:, None, None] * eye3[None, :, :]

    # bins to include: positive m_eff and finite data
    use = (m_eff > 0) & np.all(np.isfinite(S_sym.reshape(Nf, -1)), axis=1)
    if not np.any(use):
        return 0.0
    Sigma_u = Sigma[use]
    S_u = S_sym[use]
    m_u = m_eff[use]

    try:
        chol = np.linalg.cholesky(Sigma_u)                  # (Nu, 3, 3)
    except np.linalg.LinAlgError:
        return -np.inf

    diag_chol = np.real(np.diagonal(chol, axis1=1, axis2=2))
    logdet = 2.0 * np.sum(np.log(np.maximum(diag_chol, 1e-300)), axis=1)
    Sigma_inv_S = np.linalg.solve(Sigma_u, S_u)
    tr = np.real(np.trace(Sigma_inv_S, axis1=1, axis2=2))

    lnL = -float(np.sum(m_u * (logdet + tr)))
    return lnL


def loglike_et_qif_grad(
    alpha_val: float,
    rho_coeffs: np.ndarray,
    P_coeffs: np.ndarray,
    B_real_coeffs: np.ndarray,
    B_imag_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    L: float,
    c0: float,
    lP: float,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    T_eff_sq: np.ndarray | None = None,
    T_over_f2_avg: np.ndarray | None = None,
    T_over_fgamma_avg: np.ndarray | None = None,
    P_floor: np.ndarray | None = None,
    B_clip: np.ndarray | None = None,
    eps: float = 1e-9,
    jitter_floor: float = 1e-50,
    clip_logP: float = 700.0,
    clip_logit_rho: float = 50.0,
    clip_log_alpha: float = 700.0,
    B_elem_clip_factor: float = 1e6,
    Phi: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Log-likelihood AND closed-form analytic gradient w.r.t. all spline
    coefficients and log-alpha.

    Finite-difference gradients of a lnL ~ 1e8 surface carry cancellation
    noise of order f*eps_mach/h ~ 0.4 per component; that noise floor is what
    limited the v0.3 profile upper limit. The Wishart score is available in
    closed form:

        d lnL / d theta = sum_k tr[ W_k  dSigma_k/dtheta ],
        W_k = m_k ( Sigma_k^{-1} S_hat_k Sigma_k^{-1} - Sigma_k^{-1} ),

    chained through the model construction (splines, sigmoid/exp links,
    rank-r factor, phase congruence, jitter). Clips and floors contribute
    zero gradient where active (they are overflow guards, not priors).

    The forward pass mirrors loglike_et_qif operation-for-operation, so the
    returned lnL agrees with it to double precision.

    Returns: (lnL, grads) with grads keyed like the params dict; grads are
    d lnL/d(coefficient) in the same shapes as the coefficient arrays, plus
    the scalar "alpha_val" entry (zero when alpha_val = -inf).
    """
    Nf = S_hat.shape[0]
    n_coeff = P_coeffs.shape[0]
    r = B_real_coeffs.shape[2]

    def _zero_grads() -> dict:
        return {
            "alpha_val": 0.0,
            "rho_coeffs": np.zeros((n_coeff, 1)),
            "P_coeffs": np.zeros((n_coeff, 3)),
            "B_real_coeffs": np.zeros((n_coeff, 3, r)),
            "B_imag_coeffs": np.zeros((n_coeff, 3, r)),
            "phi_coeffs": np.zeros((n_coeff, 2)),
        }

    # --- splines (forward identical to loglike_et_qif) ---
    P_log = evaluate_spline(P_coeffs, knots, freqs)
    if np.any(~np.isfinite(P_log)):
        return -np.inf, _zero_grads()
    P_exp = np.exp(np.clip(P_log, -clip_logP, clip_logP))
    if P_floor is not None:
        P = np.maximum(P_exp, P_floor)
    else:
        P = np.maximum(P_exp, jitter_floor)
    dP_dlog = np.where((np.abs(P_log) < clip_logP) & (P_exp >= P), P_exp, 0.0)

    rho_logit = evaluate_spline(rho_coeffs, knots, freqs).reshape(-1)
    if np.any(~np.isfinite(rho_logit)):
        return -np.inf, _zero_grads()
    rho_pre = 1.0 / (1.0 + np.exp(-np.clip(rho_logit, -clip_logit_rho, clip_logit_rho)))
    rho = np.clip(rho_pre, 1e-6, 1.0 - 1e-6)
    drho_dlogit = np.where(
        (np.abs(rho_logit) < clip_logit_rho) & (rho_pre > 1e-6) & (rho_pre < 1.0 - 1e-6),
        rho_pre * (1.0 - rho_pre), 0.0)

    B_real = evaluate_spline(B_real_coeffs, knots, freqs)
    B_imag = evaluate_spline(B_imag_coeffs, knots, freqs)
    if np.any(~np.isfinite(B_real)) or np.any(~np.isfinite(B_imag)):
        return -np.inf, _zero_grads()

    phi_23 = evaluate_spline(phi_coeffs, knots, freqs)
    if np.any(~np.isfinite(phi_23)):
        return -np.inf, _zero_grads()
    phi_23 = wrap_phase(phi_23)

    if np.isneginf(alpha_val):
        alpha = 0.0
        alpha_active = False
    else:
        alpha = float(np.exp(np.clip(alpha_val, -clip_log_alpha, clip_log_alpha)))
        alpha_active = abs(alpha_val) < clip_log_alpha

    if alpha > 0:
        S_path = qif_path_psd_binavg(
            alpha, c0, lP, L, f_min, f_max,
            T_eff_sq=T_eff_sq, T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma, use_planck_scale=use_planck_scale)
    else:
        S_path = np.zeros(Nf, dtype=float)

    # elementwise clip (overflow guard) with gradient mask
    if B_clip is not None:
        B_elem_clip = B_elem_clip_factor * np.sqrt(np.maximum(B_clip, jitter_floor))
        B_elem_clip = B_elem_clip.reshape(Nf, 1, 1)
        mask_Br = (np.abs(B_real) < B_elem_clip).astype(float)
        mask_Bi = (np.abs(B_imag) < B_elem_clip).astype(float)
        B_real = np.clip(B_real, -B_elem_clip, B_elem_clip)
        B_imag = np.clip(B_imag, -B_elem_clip, B_elem_clip)
    else:
        mask_Br = np.ones_like(B_real)
        mask_Bi = np.ones_like(B_imag)

    eye3 = np.eye(3, dtype=complex)
    B = (B_real + 1j * B_imag).astype(complex)                  # (Nf, 3, r)
    scale = np.ones(Nf)
    need = np.zeros(Nf, dtype=bool)
    dmax = np.zeros(Nf)
    jmax_B = np.zeros(Nf, dtype=int)
    if B_clip is not None:
        diag_B = np.sum(np.abs(B) ** 2, axis=2)                 # (Nf, 3)
        dmax = np.max(diag_B, axis=1)
        jmax_B = np.argmax(diag_B, axis=1)
        limit = np.asarray(B_clip, dtype=float)
        need = np.isfinite(dmax) & np.isfinite(limit) & (dmax > limit) & (dmax > 0.0) & (limit > 0.0)
        scale[need] = np.sqrt(limit[need] / dmax[need])
    Bs = B * scale[:, None, None]
    Sigma_env = Bs @ np.conjugate(np.transpose(Bs, (0, 2, 1)))

    Sigma = Sigma_env.copy()
    Sigma[:, 0, 0] += P[:, 0]
    Sigma[:, 1, 1] += P[:, 1]
    Sigma[:, 2, 2] += P[:, 2]

    if alpha > 0:
        J_off = np.ones((3, 3)) - np.eye(3)
        M = 2.0 * np.eye(3)[None, :, :] - rho[:, None, None] * J_off[None, :, :]
        Sigma = Sigma + (S_path[:, None, None] * M).astype(complex)

    g = np.stack([np.ones(Nf, dtype=complex),
                  np.exp(1j * phi_23[:, 0]),
                  np.exp(1j * phi_23[:, 1])], axis=1)           # (Nf, 3)
    Sigma = Sigma * (g[:, :, None] * np.conjugate(g[:, None, :]))

    Sigma = 0.5 * (Sigma + np.conjugate(np.transpose(Sigma, (0, 2, 1))))
    Sigma_ph = Sigma.copy()                                     # pre-jitter, post-phase
    S_sym = 0.5 * (S_hat + np.conjugate(np.transpose(S_hat, (0, 2, 1))))

    diag_now = np.real(np.diagonal(Sigma, axis1=1, axis2=2))
    max_diag = np.max(diag_now, axis=1)
    jmax_d = np.argmax(diag_now, axis=1)
    Sigma = Sigma + (eps * max_diag + jitter_floor)[:, None, None] * eye3[None, :, :]

    use = (m_eff > 0) & np.all(np.isfinite(S_sym.reshape(Nf, -1)), axis=1)
    if not np.any(use):
        return 0.0, _zero_grads()
    idx = np.where(use)[0]
    Sigma_u = Sigma[idx]
    S_u = S_sym[idx]
    m_u = m_eff[idx]

    try:
        chol = np.linalg.cholesky(Sigma_u)
    except np.linalg.LinAlgError:
        return -np.inf, _zero_grads()

    diag_chol = np.real(np.diagonal(chol, axis1=1, axis2=2))
    logdet = 2.0 * np.sum(np.log(np.maximum(diag_chol, 1e-300)), axis=1)
    Sigma_inv_S = np.linalg.solve(Sigma_u, S_u)
    tr = np.real(np.trace(Sigma_inv_S, axis1=1, axis2=2))
    lnL = -float(np.sum(m_u * (logdet + tr)))

    # ---- analytic score ----
    Sigma_inv = np.linalg.inv(Sigma_u)
    W_u = m_u[:, None, None] * (Sigma_inv @ S_u @ Sigma_inv - Sigma_inv)
    W = np.zeros((Nf, 3, 3), dtype=complex)
    W[idx] = W_u

    # phases: d lnL/d phi_a = 2 Im[(W Sigma_ph)_aa]   (jitter is phase-invariant)
    WSph = W @ Sigma_ph
    dphi = 2.0 * np.stack([np.imag(WSph[:, 1, 1]), np.imag(WSph[:, 2, 2])], axis=1)

    # inner-parameter effective weight: G~ = D^H W D, plus the jitter chain
    # eps * tr(W) on the argmax-diagonal entry.
    Gt = W * (np.conjugate(g)[:, :, None] * g[:, None, :])
    trW = np.real(np.trace(W, axis1=1, axis2=2))
    G_eff = Gt.copy()
    G_eff[np.arange(Nf), jmax_d, jmax_d] += eps * trW

    # instrument PSDs (log parameterization)
    gP = np.real(np.diagonal(G_eff, axis1=1, axis2=2)) * dP_dlog        # (Nf, 3)

    # rho and alpha
    if alpha > 0:
        trG = np.real(np.trace(G_eff, axis1=1, axis2=2))
        off_sum = np.real(np.sum(G_eff, axis=(1, 2))) - trG
        grho = -S_path * off_sum * drho_dlogit
        galpha = float(np.sum(S_path * (2.0 * trG - rho * off_sum))) if alpha_active else 0.0
    else:
        grho = np.zeros(Nf)
        galpha = 0.0

    # environmental factor: Sigma_env = (sB)(sB)^H with s the rescale guard
    GB = G_eff @ Bs                                                     # (Nf, 3, r)
    gBr = 2.0 * scale[:, None, None] * np.real(GB)
    gBi = 2.0 * scale[:, None, None] * np.imag(GB)
    if np.any(need):
        t0 = np.real(np.einsum('kij,kji->k', G_eff, Sigma_env))
        for k in np.where(need)[0]:
            j = jmax_B[k]
            gBr[k, j, :] += t0[k] * (-2.0 * np.real(B[k, j, :]) / dmax[k])
            gBi[k, j, :] += t0[k] * (-2.0 * np.imag(B[k, j, :]) / dmax[k])
    gBr *= mask_Br
    gBi *= mask_Bi

    # chain through the B-spline design matrix
    if Phi is None:
        Phi = _spline_design_matrix(knots, freqs, n_coeff)
    grads = {
        "alpha_val": galpha,
        "rho_coeffs": (Phi.T @ grho).reshape(n_coeff, 1),
        "P_coeffs": Phi.T @ gP,
        "B_real_coeffs": np.einsum('kn,kjl->njl', Phi, gBr),
        "B_imag_coeffs": np.einsum('kn,kjl->njl', Phi, gBi),
        "phi_coeffs": Phi.T @ dphi,
    }
    return lnL, grads


# ---------------------------
# Bootstrap primitives (moment-matching for fractional m_eff)
# ---------------------------

def draw_m_star(m_eff: float, rng: np.random.Generator) -> int:
    """
    Probabilistic rounding for fractional effective sample sizes:
      m0 = floor(m), p = m-m0
      m* = m0+1 w.p. p else m0
    """
    if not np.isfinite(m_eff) or m_eff <= 0:
        return 0
    m0 = int(np.floor(m_eff))
    p = float(m_eff - m0)
    return (m0 + 1) if (rng.random() < p) else m0


def sample_covariance_from_sigma(Sigma: np.ndarray, m_star: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate sample covariance S_hat = (1/m_star) Σ_j y_j y_j^H
    where y_j ~ CN(0, Sigma), using Cholesky sampling.

    Returns: (3,3) complex Hermitian.
    """
    if m_star <= 0:
        return np.full((3, 3), np.nan, dtype=complex)

    Sigma = hermitize(Sigma)
    try:
        Lc = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        max_diag = float(np.max(np.real(np.diag(Sigma))))
        Sigma2 = Sigma + (1e-12 * max_diag + 1e-50) * np.eye(3)
        Lc = np.linalg.cholesky(Sigma2)

    z1 = rng.standard_normal((3, m_star))
    z2 = rng.standard_normal((3, m_star))
    z = (z1 + 1j * z2) / np.sqrt(2.0)
    y = Lc @ z
    S = (y @ y.conj().T) / float(m_star)
    return hermitize(S)


# ---------------------------
# Sign-channel statistic (gauge-invariant, fit-free)
# ---------------------------

def sign_channel_stat(S_hat: np.ndarray, m_eff: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Fit-free detection statistic built on the gauge-invariant triple product
        t(f) = S12(f) S23(f) S31(f).

    For any diagonal + rank-1 + phase-calibration model covariance,
    Re t >= 0 (identifiability theorem); the geodesic template gives
    Re t = -(rho S_path)^3 < 0. The statistic is

        T = sum_k m_k^{3/2} r_k,   r_k = Re t_k / (S11 S22 S33)_k,

    where r_k is the normalized triple coherence (dimensionless) and the
    m^{3/2} weight equalizes per-bin null variance (each off-diagonal
    coherence fluctuates as 1/sqrt(m), so r_k ~ m^{-3/2} under the null).

    t is invariant under per-channel phases AND per-channel sign flips
    (each channel enters the product exactly twice), so T does not depend
    on the Michelson arm-orientation convention of the input channels.

    Returns (T, r) with r the per-bin normalized triple coherence
    (NaN on excluded bins).
    """
    Nf = S_hat.shape[0]
    S_sym = 0.5 * (S_hat + np.conjugate(np.transpose(S_hat, (0, 2, 1))))
    d = np.real(S_sym[:, 0, 0] * S_sym[:, 1, 1] * S_sym[:, 2, 2])
    t = S_sym[:, 0, 1] * S_sym[:, 1, 2] * S_sym[:, 2, 0]
    use = (m_eff > 0) & np.all(np.isfinite(S_sym.reshape(Nf, -1)), axis=1) & (d > 0)
    r = np.full(Nf, np.nan)
    r[use] = np.real(t[use]) / d[use]
    T = float(np.sum(np.asarray(m_eff, dtype=float)[use] ** 1.5 * r[use]))
    return T, r


def sample_wishart_bartlett(chol_stack: np.ndarray, m_int: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Exact complex-Wishart sample covariances S ~ (1/m) CW_3(m, Sigma) for a
    stack of bins, via the Bartlett decomposition: O(1) per bin regardless
    of m (direct outer-product sampling costs O(m) and is prohibitive at the
    m_eff ~ 1e3 of high-frequency log-averaged bins).

    chol_stack: (Nf, 3, 3) Cholesky factors of the Sigma stack.
    m_int:      (Nf,) integer sample counts (>= 1).

    S = L (A A^H / m) L^H with A lower-triangular, |A_jj|^2 ~ Gamma(m-j, 1)
    (complex Wishart) and A_ij ~ CN(0, 1) for i > j.
    """
    Nf = chol_stack.shape[0]
    m = np.asarray(m_int, dtype=float)
    A = np.zeros((Nf, 3, 3), dtype=complex)
    for j in range(3):
        # complex Wishart: squared diagonal ~ Gamma(m - j, 1)
        shape = np.maximum(m - j, 1e-8)
        A[:, j, j] = np.sqrt(rng.gamma(shape=shape, scale=1.0))
        for i in range(j + 1, 3):
            A[:, i, j] = (rng.standard_normal(Nf) + 1j * rng.standard_normal(Nf)) / np.sqrt(2.0)
    LA = chol_stack @ A
    S = (LA @ np.conjugate(np.transpose(LA, (0, 2, 1)))) / m[:, None, None]
    return S


def sign_channel_pvalue(
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    Sigma_null: np.ndarray,
    n_boot: int = 400,
    seed: int | None = None,
    diagonalize_null: bool = True,
) -> tuple[float, float, np.ndarray]:
    """
    One-sided p-value of the sign-channel statistic against a parametric
    bootstrap under a null model stack Sigma_null (normally the fitted
    H_env covariance): p = P(T* <= T_obs). Small p = the data's triple
    product is more negative than the environmental model class can
    produce.

    diagonalize_null (default True): calibrate under the DIAGONAL of the
    fitted null rather than the full fitted covariance. The fitted rank-1
    factor soaks up the draw's own off-diagonal sampling noise, and any
    diag+rank-1 structure biases the model-level triple product positive
    (the sign theorem), so a plug-in calibration under the full fitted
    model shifts the bootstrap distribution positive and is
    anti-conservative (measured: false-detection rate 0.56 at nominal 0.05
    on true-null draws). Under the diagonal null the calibration is exact
    for independent channels and, by the same theorem, conservative in the
    presence of any genuine admissible environmental coherence (which can
    only push the observed statistic positive, away from the signal side).

    Returns (T_obs, p, T_boot).
    """
    rng = np.random.default_rng(seed)
    T_obs, _ = sign_channel_stat(S_hat, m_eff)
    # fractional m_eff is passed through directly: the Bartlett construction
    # generalizes to non-integer sample counts (Gamma shapes m, m-1, m-2),
    # matching the fractional-Wishart moments the m_eff prescription implies.
    # Rounding (the previous behavior) biases the null width by O(1/2m) per
    # bin -- negligible at m >~ 70 but wrong in principle.
    m_frac = np.maximum(np.asarray(m_eff, dtype=float), 1.0)
    Sig = np.asarray(Sigma_null, dtype=complex)
    Sig = 0.5 * (Sig + np.conjugate(np.transpose(Sig, (0, 2, 1))))
    if diagonalize_null:
        d = np.real(np.diagonal(Sig, axis1=1, axis2=2))
        Sig = d[:, :, None] * np.eye(3)[None, :, :].astype(complex)
    max_diag = np.max(np.real(np.diagonal(Sig, axis1=1, axis2=2)), axis=1)
    Sig = Sig + (1e-12 * max_diag + 1e-50)[:, None, None] * np.eye(3)[None, :, :]
    chol = np.linalg.cholesky(Sig)
    T_boot = np.empty(n_boot)
    for b in range(n_boot):
        S_b = sample_wishart_bartlett(chol, m_frac, rng)
        T_boot[b], _ = sign_channel_stat(S_b, m_eff)
    p = float((np.sum(T_boot <= T_obs) + 1.0) / (n_boot + 1.0))
    return T_obs, p, T_boot


# ---------------------------
# Simple helpers (from legacy script) for knots + synthetic demo
# ---------------------------

def _open_uniform_knots(x_min: float, x_max: float, n_coeff: int, k: int = 3) -> np.ndarray:
    if n_coeff < k + 1:
        raise ValueError("n_coeff must be at least k+1")
    n_interior = n_coeff - k - 1
    if n_interior > 0:
        interior = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
        knots = np.concatenate([np.full(k + 1, x_min), interior, np.full(k + 1, x_max)])
    else:
        knots = np.concatenate([np.full(k + 1, x_min), np.full(k + 1, x_max)])
    return knots


def _open_log_knots(x_min: float, x_max: float, n_coeff: int, k: int = 3) -> np.ndarray:
    """Open knot vector with geometrically spaced interior knots.

    For spectra analyzed on log-spaced bins, uniform-in-frequency knots put
    almost no resolution in the lowest decade (where an f^-gamma signal
    lives); geometric spacing matches the bin density.
    """
    if x_min <= 0:
        return _open_uniform_knots(x_min, x_max, n_coeff, k)
    if n_coeff < k + 1:
        raise ValueError("n_coeff must be at least k+1")
    n_interior = n_coeff - k - 1
    if n_interior > 0:
        interior = np.geomspace(x_min, x_max, n_interior + 2)[1:-1]
        knots = np.concatenate([np.full(k + 1, x_min), interior, np.full(k + 1, x_max)])
    else:
        knots = np.concatenate([np.full(k + 1, x_min), np.full(k + 1, x_max)])
    return knots


def _make_synthetic_data(n_f: int = 8, n_coeff: int = 6, r: int = 1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    f_lo = 10.0
    f_hi = 100.0
    freqs = np.linspace(f_lo + 1.0, f_hi - 1.0, n_f)
    df = 1.0
    f_min = freqs - 0.5 * df
    f_max = freqs + 0.5 * df

    knots = _open_uniform_knots(f_lo, f_hi, n_coeff, k=3)

    P_coeffs = np.full((n_coeff, 3), np.log(1e-20))
    rho_coeffs = np.full((n_coeff, 1), 0.0)
    B_real_coeffs = np.full((n_coeff, 3, r), 1e-22)
    B_imag_coeffs = np.zeros((n_coeff, 3, r))
    phi_coeffs = np.zeros((n_coeff, 2))

    A = rng.normal(size=(n_f, 3, 3)) + 1j * rng.normal(size=(n_f, 3, 3))
    S_hat = A @ np.conjugate(np.transpose(A, (0, 2, 1)))
    scale = 1e-20 / np.mean(np.real(np.diagonal(S_hat, axis1=1, axis2=2)))
    S_hat *= scale

    m_eff = np.full((n_f,), 10.0)
    T_over_f2_avg = 1.0 / (f_min * f_max)

    P_floor = np.full((n_f, 3), 1e-22)
    B_clip = np.full((n_f,), 1e-18)

    return {
        "alpha_val": np.log(1.0),
        "rho_coeffs": rho_coeffs,
        "P_coeffs": P_coeffs,
        "B_real_coeffs": B_real_coeffs,
        "B_imag_coeffs": B_imag_coeffs,
        "phi_coeffs": phi_coeffs,
        "S_hat": S_hat,
        "m_eff": m_eff,
        "freqs": freqs,
        "f_min": f_min,
        "f_max": f_max,
        "knots": knots,
        "L": 1.0,
        "c0": 1.0,
        "lP": 1.0,
        "T_over_f2_avg": T_over_f2_avg,
        "P_floor": P_floor,
        "B_clip": B_clip,
    }


# ---------------------------
# GWF loading + CSD construction
# ---------------------------

_GWF_RE = re.compile(r"E-(E[0-3])_STRAIN-(\d+)-(\d+)\.gwf$")


def _parse_gwf_filename(path: str) -> tuple[str, int, int] | None:
    base = os.path.basename(path)
    match = _GWF_RE.match(base)
    if not match:
        return None
    channel, gps_str, dur_str = match.groups()
    return channel, int(gps_str), int(dur_str)


def _load_line_mask(path: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            try:
                f_lo = float(parts[0])
                f_hi = float(parts[1])
            except ValueError:
                continue
            if f_hi < f_lo:
                f_lo, f_hi = f_hi, f_lo
            ranges.append((f_lo, f_hi))
    return ranges


def _load_transfer_csv(path: str, is_sq: bool) -> tuple[np.ndarray, np.ndarray]:
    freqs = []
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            f_hz = float(parts[0])
            if len(parts) >= 3:
                re_val = float(parts[1])
                im_val = float(parts[2])
                t_sq = re_val * re_val + im_val * im_val
            else:
                mag = float(parts[1])
                t_sq = mag if is_sq else mag * mag
            freqs.append(f_hz)
            vals.append(t_sq)
    if not freqs:
        raise ValueError(f"No transfer data read from {path}")
    f_arr = np.array(freqs, dtype=float)
    t_arr = np.array(vals, dtype=float)
    order = np.argsort(f_arr)
    return f_arr[order], t_arr[order]


def _transfer_sq_func(freqs: np.ndarray, f_src: np.ndarray, t_sq_src: np.ndarray) -> np.ndarray:
    return np.interp(freqs, f_src, t_sq_src, left=t_sq_src[0], right=t_sq_src[-1])


def _find_gwf_groups(data_root: str) -> list[dict]:
    groups: dict[tuple[str, int], dict] = {}
    for root, _, files in os.walk(data_root):
        rel_root = os.path.relpath(root, data_root)
        set_name = "" if rel_root == "." else rel_root.split(os.sep)[0]
        for name in files:
            if not name.endswith(".gwf"):
                continue
            parsed = _parse_gwf_filename(name)
            if parsed is None:
                continue
            channel, gps, duration = parsed
            if channel not in ("E1", "E2", "E3"):
                continue
            key = (set_name, gps)
            entry = groups.setdefault(key, {"set": set_name, "gps": gps, "duration": duration, "paths": {}})
            entry["paths"][channel] = os.path.join(root, name)
    out = []
    for entry in groups.values():
        if all(ch in entry["paths"] for ch in ("E1", "E2", "E3")):
            out.append(entry)
    return sorted(out, key=lambda e: (e["set"], e["gps"]))


def _read_gwf_triplet(paths: dict, max_seconds: float) -> tuple[list[np.ndarray], float]:
    try:
        from gwpy.timeseries import TimeSeries
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("gwpy is required to read .gwf files (pip install gwpy).") from exc

    data = []
    sample_rate = None
    min_len = None
    for channel in ("E1", "E2", "E3"):
        path = paths[channel]
        parsed = _parse_gwf_filename(path)
        if parsed is None:
            raise ValueError(f"Unrecognized GWF filename: {path}")
        _, gps, duration = parsed
        seg = min(max_seconds, float(duration))
        ts = TimeSeries.read(path, f"{channel}:STRAIN", start=gps, end=gps + seg)
        series = np.asarray(ts.value, dtype=float)
        data.append(series)
        sr = float(ts.sample_rate.value)
        sample_rate = sr if sample_rate is None else sample_rate
        min_len = len(series) if min_len is None else min(min_len, len(series))
    if sample_rate is None or min_len is None:
        raise RuntimeError("Failed to read any data from GWF files.")
    data = [x[:min_len] for x in data]
    return data, sample_rate


def _compute_welch_csd_matrix(
    data: list[np.ndarray],
    fs: float,
    nperseg_seconds: float = 4.0,
    overlap: float = 0.5,
    fmin_hz: float = 10.0,
    fmax_hz: float | None = None,
    max_bins: int | None = None,
    line_mask: list[tuple[float, float]] | None = None,
    window: str = "hann",
    bin_spacing: str = "log",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Welch cross-spectral matrices on an averaged analysis-bin grid.

    When ``max_bins`` is set, native Welch bins are AVERAGED into ``max_bins``
    blocks with ``bin_spacing`` = "log" (default) or "linear" edges, and
    ``m_eff`` scales with the number of native bins per block (corrected for
    the window-induced correlation between neighboring native bins).
    ``bin_spacing`` = "subsample" reproduces the historical behavior of keeping
    every Nth native bin, which discards the data between kept bins and, for a
    linear grid over a wide band, leaves a single analysis bin below ~100 Hz --
    that starves an f^-gamma search of exactly the frequencies that carry its
    information.
    """
    n_samples = min(len(x) for x in data)
    nperseg = int(max(8, round(nperseg_seconds * fs)))
    nperseg = min(nperseg, n_samples)
    noverlap = int(max(0, min(nperseg - 1, round(nperseg * overlap))))
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("Invalid overlap; nperseg - noverlap must be > 0.")
    if n_samples < nperseg:
        raise ValueError("Not enough samples for Welch estimate.")

    nseg = 1 + (n_samples - nperseg) // step

    w = signal.get_window(window, nperseg, fftbins=True)
    U = np.sum(w ** 2)
    if U <= 0:
        raise ValueError("Window normalization is zero.")

    def _effective_m(k: int) -> float:
        if k <= 1:
            return 1.0
        corr_sum = 0.0
        for j in range(1, k):
            shift = j * step
            if shift >= nperseg:
                break
            c = float(np.sum(w[:nperseg - shift] * w[shift:]) / U)
            corr_sum += (1.0 - j / k) * (c ** 2)
        return float(k / (1.0 + 2.0 * corr_sum))

    freqs, P11 = signal.welch(data[0], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
    _, P22 = signal.welch(data[1], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
    _, P33 = signal.welch(data[2], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")

    S_hat = np.zeros((len(freqs), 3, 3), dtype=complex)
    S_hat[:, 0, 0] = P11
    S_hat[:, 1, 1] = P22
    S_hat[:, 2, 2] = P33

    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        _, Pij = signal.csd(data[i], data[j], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
        S_hat[:, i, j] = Pij
        S_hat[:, j, i] = np.conjugate(Pij)

    mask = freqs > 0
    if fmin_hz is not None:
        mask &= freqs >= fmin_hz
    if fmax_hz is not None:
        mask &= freqs <= fmax_hz

    if line_mask:
        for f_lo, f_hi in line_mask:
            mask &= ~((freqs >= f_lo) & (freqs <= f_hi))

    freqs = freqs[mask]
    S_hat = S_hat[mask]

    # Native Welch bin width; must be computed BEFORE any subsampling so bin
    # edges keep describing one Welch bin.
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    m_eff_val = _effective_m(nseg)

    if max_bins is not None and len(freqs) > max_bins and bin_spacing != "subsample":
        # Correlation between neighboring native bins induced by the window
        # (for Hann, adjacent-bin rho^2 ~ 0.44): the effective number of
        # independent native bins in a block of n is n / (1 + 2*sum(1-j/n)*rho_j^2).
        w2 = w ** 2
        W = np.abs(np.fft.rfft(w2)) / np.sum(w2)
        rho_sq = (W[1:] ** 2)
        rho_sq = rho_sq[rho_sq > 1e-6]

        def _n_indep(n: int) -> float:
            if n <= 1:
                return float(max(n, 1))
            j = np.arange(1, min(len(rho_sq), n - 1) + 1)
            corr = float(np.sum((1.0 - j / n) * rho_sq[: len(j)]))
            return n / (1.0 + 2.0 * corr)

        f_lo = float(freqs[0]) - 0.5 * df
        f_hi = float(freqs[-1]) + 0.5 * df
        if bin_spacing == "log" and f_lo > 0:
            edges = np.geomspace(f_lo, f_hi, max_bins + 1)
        else:
            edges = np.linspace(f_lo, f_hi, max_bins + 1)
        which = np.searchsorted(edges, freqs, side="right") - 1
        which = np.clip(which, 0, max_bins - 1)

        freqs_b, fmin_b, fmax_b, S_b, m_b = [], [], [], [], []
        for b in range(max_bins):
            sel = which == b
            n_nat = int(sel.sum())
            if n_nat == 0:
                continue
            f_sel = freqs[sel]
            freqs_b.append(float(np.exp(np.mean(np.log(f_sel)))) if bin_spacing == "log"
                           else float(np.mean(f_sel)))
            fmin_b.append(float(f_sel[0]) - 0.5 * df)
            fmax_b.append(float(f_sel[-1]) + 0.5 * df)
            S_b.append(np.mean(S_hat[sel], axis=0))
            m_b.append(m_eff_val * _n_indep(n_nat))
        freqs = np.asarray(freqs_b)
        f_min = np.asarray(fmin_b)
        f_max = np.asarray(fmax_b)
        S_hat = np.asarray(S_b)
        m_eff = np.asarray(m_b)
    else:
        if max_bins is not None and len(freqs) > max_bins:
            idx = np.linspace(0, len(freqs) - 1, max_bins).round().astype(int)
            freqs = freqs[idx]
            S_hat = S_hat[idx]
        m_eff = np.full_like(freqs, float(m_eff_val))
        f_min = freqs - 0.5 * df
        f_max = freqs + 0.5 * df

    valid = f_min > 0
    return freqs[valid], f_min[valid], f_max[valid], S_hat[valid], m_eff[valid]


def _spline_design_matrix(knots: np.ndarray, freqs: np.ndarray, n_coeff: int) -> np.ndarray:
    """B-spline design matrix Phi (n_f x n_coeff): Phi @ coeffs = evaluate_spline(coeffs)."""
    Phi = np.zeros((len(freqs), n_coeff))
    for j in range(n_coeff):
        e = np.zeros((n_coeff, 1))
        e[j, 0] = 1.0
        Phi[:, j] = evaluate_spline(e, knots, freqs)[:, 0]
    return Phi


def _build_initial_coeffs(S_hat: np.ndarray, n_coeff: int, r: int,
                          freqs: np.ndarray | None = None,
                          knots: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag = np.real(np.diagonal(S_hat, axis1=1, axis2=2))
    median_diag = np.maximum(np.median(diag, axis=0), 1e-300)
    P_coeffs = np.tile(np.log(median_diag), (n_coeff, 1))
    if freqs is not None and knots is not None:
        # Least-squares spline fit of the log periodogram per channel: a flat
        # start is many e-folds from the true PSD shape and needlessly hard
        # for a 200+-dimensional finite-difference optimization.
        try:
            Phi = _spline_design_matrix(knots, freqs, n_coeff)
            y = np.log(np.maximum(diag, 1e-300))
            good = np.all(np.isfinite(y), axis=1)
            if int(good.sum()) >= n_coeff:
                sol, *_ = np.linalg.lstsq(Phi[good], y[good], rcond=None)
                if np.all(np.isfinite(sol)):
                    P_coeffs = sol
        except Exception:
            pass
    rho_coeffs = np.zeros((n_coeff, 1))
    B_real_coeffs = np.zeros((n_coeff, 3, r))
    B_imag_coeffs = np.zeros((n_coeff, 3, r))
    try:
        S_mean = hermitize(np.mean(S_hat, axis=0))
        S_env = hermitize(S_mean - np.diag(median_diag))
        w, v = np.linalg.eigh(S_env)
        order = np.argsort(w)[::-1]
        w = w[order]
        v = v[:, order]
        B0 = np.zeros((3, r), dtype=complex)
        for i in range(min(r, len(w))):
            eig = float(np.real(w[i]))
            if eig > 0:
                B0[:, i] = np.sqrt(eig) * v[:, i]
        B_real_coeffs = np.tile(B0.real, (n_coeff, 1, 1))
        B_imag_coeffs = np.tile(B0.imag, (n_coeff, 1, 1))
    except Exception:
        pass
    phi_coeffs = np.zeros((n_coeff, 2))
    return rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs


def _pack_params(
    alpha_val: float,
    rho_coeffs: np.ndarray,
    P_coeffs: np.ndarray,
    B_real_coeffs: np.ndarray,
    B_imag_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    fit_alpha: bool,
    fit_phi: bool,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if fit_alpha:
        parts.append(np.array([alpha_val], dtype=float))
    parts.append(rho_coeffs.reshape(-1))
    parts.append(P_coeffs.reshape(-1))
    parts.append(B_real_coeffs.reshape(-1))
    parts.append(B_imag_coeffs.reshape(-1))
    if fit_phi:
        parts.append(phi_coeffs.reshape(-1))
    return np.concatenate(parts)


def _unpack_params(
    x: np.ndarray,
    n_coeff: int,
    r: int,
    fit_alpha: bool,
    fit_phi: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offset = 0
    if fit_alpha:
        alpha_val = float(x[offset])
        offset += 1
    else:
        alpha_val = float("-inf")

    rho_coeffs = x[offset:offset + n_coeff].reshape(n_coeff, 1)
    offset += n_coeff

    P_coeffs = x[offset:offset + n_coeff * 3].reshape(n_coeff, 3)
    offset += n_coeff * 3

    B_real_coeffs = x[offset:offset + n_coeff * 3 * r].reshape(n_coeff, 3, r)
    offset += n_coeff * 3 * r

    B_imag_coeffs = x[offset:offset + n_coeff * 3 * r].reshape(n_coeff, 3, r)
    offset += n_coeff * 3 * r

    if fit_phi:
        phi_coeffs = x[offset:offset + n_coeff * 2].reshape(n_coeff, 2)
    else:
        phi_coeffs = np.zeros((n_coeff, 2))

    return alpha_val, rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs


def _default_bounds(
    n_coeff: int,
    r: int,
    fit_alpha: bool,
    fit_phi: bool,
    P_log_center: float = 0.0,
    alpha_log_center: float = 0.0,
    B_scale: float = 1.0,
) -> list[tuple[float, float] | None]:
    """
    Box bounds centered on the DATA scale. Strain data live at log P ~ -108;
    fixed bounds like (-30, 30) pin every parameter at a corner and make the
    fit data-independent (the identical-LR pathology).
    """
    bounds: list[tuple[float, float] | None] = []
    if fit_alpha:
        # exp(-35) relative to the data scale is numerically "signal off",
        # so the nested null (alpha = 0) is inside the box to high accuracy.
        bounds.append((alpha_log_center - 35.0, alpha_log_center + 15.0))
    bounds.extend([(-10.0, 10.0)] * n_coeff)           # rho logits
    bounds.extend([(P_log_center - 25.0, P_log_center + 25.0)] * (n_coeff * 3))  # log P
    B_lim = 1e3 * max(B_scale, 1e-300)
    bounds.extend([(-B_lim, B_lim)] * (n_coeff * 3 * r))    # B real
    bounds.extend([(-B_lim, B_lim)] * (n_coeff * 3 * r))    # B imag
    if fit_phi:
        bounds.extend([(-np.pi, np.pi)] * (n_coeff * 2))
    return bounds


def _data_scales(S_hat: np.ndarray, freqs: np.ndarray, gamma: float,
                 use_planck_scale: bool, L: float, c0: float, lP: float) -> tuple[float, float, float]:
    """
    Return (P_log_center, alpha_log_center, B_scale) inferred from the data.

    alpha is scaled so that S_path = alpha * f^-gamma matches the median PSD at
    the band center: alpha ~ P_med * f_med^gamma.
    """
    diag = np.real(np.diagonal(S_hat, axis1=1, axis2=2))
    diag = diag[np.isfinite(diag)]
    P_scale = float(np.median(np.maximum(diag, 1e-300))) if diag.size else 1.0
    P_log_center = float(np.log(P_scale))
    f_med = float(np.median(freqs)) if len(freqs) else 1.0
    alpha_log_center = P_log_center + gamma * float(np.log(max(f_med, 1e-300)))
    if use_planck_scale:
        pref = (c0 * lP) / (2.0 * np.pi ** 2 * L ** 2)
        alpha_log_center -= float(np.log(max(pref, 1e-300)))
    B_scale = float(np.sqrt(P_scale))
    return P_log_center, alpha_log_center, B_scale


def _clip_to_bounds(x: np.ndarray, bounds: list[tuple[float, float] | None]) -> np.ndarray:
    out = x.copy()
    for i, b in enumerate(bounds):
        if b is None:
            continue
        lo, hi = b
        out[i] = min(max(out[i], lo), hi)
    return out


def fit_model(
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    P_floor: np.ndarray,
    B_clip: np.ndarray,
    n_coeff: int,
    r: int,
    L: float,
    c0: float,
    lP: float,
    T_over_f2_avg: np.ndarray,
    T_over_fgamma_avg: np.ndarray | None = None,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    fit_alpha: bool = True,
    fit_phi: bool = True,
    max_iter: int = 200,
    n_starts: int = 3,
    seed: int | None = None,
    init_params: dict | None = None,
    use_grad: bool = True,
) -> tuple[dict, float]:
    rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs = _build_initial_coeffs(
        S_hat, n_coeff, r, freqs=freqs, knots=knots)
    if init_params is not None:
        # Warm start (e.g. the alternative fit from the null optimum). With the
        # signal initialized "off", the H1 optimizer starts at the H0 optimum and
        # can only improve, so the nested LR is >= 0 up to optimizer tolerance.
        rho_coeffs = np.asarray(init_params["rho_coeffs"], dtype=float).copy()
        P_coeffs = np.asarray(init_params["P_coeffs"], dtype=float).copy()
        B_real_coeffs = np.asarray(init_params["B_real_coeffs"], dtype=float).copy()
        B_imag_coeffs = np.asarray(init_params["B_imag_coeffs"], dtype=float).copy()
        phi_coeffs = np.asarray(init_params["phi_coeffs"], dtype=float).copy()

    P_log_center, alpha_log_center, B_scale = _data_scales(
        S_hat, freqs, gamma, use_planck_scale, L, c0, lP
    )
    alpha_lo = alpha_log_center - 35.0
    alpha_val = alpha_lo if fit_alpha else float("-inf")
    if (
        init_params is not None
        and fit_alpha
        and np.isfinite(init_params.get("alpha_val", float("-inf")))
    ):
        alpha_val = float(np.clip(init_params["alpha_val"], alpha_lo, alpha_log_center + 15.0))

    bounds = _default_bounds(
        n_coeff, r, fit_alpha, fit_phi,
        P_log_center=P_log_center,
        alpha_log_center=alpha_log_center,
        B_scale=B_scale,
    )

    # Per-parameter finite-difference steps. scipy's default eps (~1.5e-8) is an
    # ABSOLUTE step: applied to B coefficients of physical scale ~1e-24 it
    # inflates B B^H by ~32 orders of magnitude, the numerical gradient is pure
    # garbage (~1e24), every line search fails, and L-BFGS-B exits at nit=0
    # with the start point untouched. Steps must be matched to each block's scale.
    eps_parts: list[np.ndarray] = []
    if fit_alpha:
        eps_parts.append(np.full(1, 1e-4))
    eps_parts.append(np.full(n_coeff, 1e-4))                       # rho logits
    eps_parts.append(np.full(n_coeff * 3, 1e-4))                   # log P
    eps_B = 1e-4 * max(B_scale, 1e-300)
    eps_parts.append(np.full(n_coeff * 3 * r, eps_B))              # B real
    eps_parts.append(np.full(n_coeff * 3 * r, eps_B))              # B imag
    if fit_phi:
        eps_parts.append(np.full(n_coeff * 2, 1e-4))               # phases
    eps_vec = np.concatenate(eps_parts)

    def objective(x: np.ndarray) -> float:
        alpha_val_i, rho_c, P_c, B_r, B_i, phi_c = _unpack_params(x, n_coeff, r, fit_alpha, fit_phi)
        lnL = loglike_et_qif(
            alpha_val_i,
            rho_c,
            P_c,
            B_r,
            B_i,
            phi_c,
            S_hat,
            m_eff,
            freqs,
            f_min,
            f_max,
            knots,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            P_floor=P_floor,
            B_clip=B_clip,
        )
        return -float(lnL)

    # Analytic-score objective (closed-form Wishart gradient). This removes
    # the finite-difference cancellation-noise floor (~0.4/component at
    # lnL ~ 1e8) that limited profile-likelihood resolution in v0.3.
    #
    # The fit runs in SCALED variables x = s * y (s = 1 for log/logit/phase
    # blocks, s = B_scale for the environmental factor blocks). Raw parameters
    # span ~26 orders of magnitude; against L-BFGS-B's identity initial
    # Hessian that conditioning makes descent from distant starts crawl.
    Phi_mat = _spline_design_matrix(knots, freqs, n_coeff) if use_grad else None
    s_parts: list[np.ndarray] = []
    if fit_alpha:
        s_parts.append(np.ones(1))
    s_parts.append(np.ones(n_coeff))                                # rho logits
    s_parts.append(np.ones(n_coeff * 3))                            # log P
    s_parts.append(np.full(n_coeff * 3 * r, max(B_scale, 1e-300)))  # B real
    s_parts.append(np.full(n_coeff * 3 * r, max(B_scale, 1e-300)))  # B imag
    if fit_phi:
        s_parts.append(np.ones(n_coeff * 2))                        # phases
    s_vec = np.concatenate(s_parts)

    def objective_with_grad_scaled(y: np.ndarray) -> tuple[float, np.ndarray]:
        x = s_vec * y
        alpha_val_i, rho_c, P_c, B_r, B_i, phi_c = _unpack_params(x, n_coeff, r, fit_alpha, fit_phi)
        lnL, gd = loglike_et_qif_grad(
            alpha_val_i, rho_c, P_c, B_r, B_i, phi_c,
            S_hat, m_eff, freqs, f_min, f_max, knots,
            L=L, c0=c0, lP=lP, gamma=gamma, use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg,
            P_floor=P_floor, B_clip=B_clip, Phi=Phi_mat,
        )
        if not np.isfinite(lnL):
            return np.inf, np.zeros_like(y)
        grad = _pack_params(gd["alpha_val"], gd["rho_coeffs"], gd["P_coeffs"],
                            gd["B_real_coeffs"], gd["B_imag_coeffs"], gd["phi_coeffs"],
                            fit_alpha, fit_phi)
        return -float(lnL), -(grad * s_vec)

    # When fitting alpha, seed the signal start with a coarse 1-D profile scan
    # over log-alpha (nuisance parameters held at the warm-start values). In
    # log-parameterization the numerical gradient w.r.t. alpha_val vanishes at
    # alpha ~ 0, and a start at the raw data scale can overshoot the basin, so
    # a gradient-free scan is the only reliable way to locate the alpha basin.
    alpha_seed = alpha_log_center
    if fit_alpha:
        scan = np.linspace(alpha_lo, alpha_log_center + 10.0, 46)
        best_scan_val = -np.inf
        for a_try in scan:
            x_try = _pack_params(float(a_try), rho_coeffs, P_coeffs, B_real_coeffs,
                                 B_imag_coeffs, phi_coeffs, fit_alpha, fit_phi)
            v = -objective(_clip_to_bounds(x_try, bounds))
            if v > best_scan_val:
                best_scan_val = v
                alpha_seed = float(a_try)

    rng = np.random.default_rng(seed)
    best_res = None
    best_fun = np.inf
    # Two deterministic starts are always used when fitting alpha: signal off
    # (nested-null warm start, guarantees LR >= 0) and the profile-scan seed.
    n_total = max(1, n_starts, 2 if fit_alpha else 1)
    for start in range(n_total):
        if start == 0:
            x0 = _pack_params(alpha_val, rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs, fit_alpha, fit_phi)
        elif start == 1 and fit_alpha:
            x0 = _pack_params(alpha_seed, rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs, fit_alpha, fit_phi)
        else:
            # Jitter relative to each parameter's physical scale; absolute
            # jitters (e.g. 1e-3 on B ~ 1e-24) would swamp the initial point.
            rho_j = rho_coeffs + rng.normal(scale=0.1, size=rho_coeffs.shape)
            P_j = P_coeffs + rng.normal(scale=0.5, size=P_coeffs.shape)
            B_jit = 0.1 * max(float(np.max(np.abs(B_real_coeffs))), float(np.max(np.abs(B_imag_coeffs))), 1e-3 * B_scale)
            B_rj = B_real_coeffs + rng.normal(scale=B_jit, size=B_real_coeffs.shape)
            B_ij = B_imag_coeffs + rng.normal(scale=B_jit, size=B_imag_coeffs.shape)
            phi_j = phi_coeffs + rng.normal(scale=0.1, size=phi_coeffs.shape)
            alpha_j = alpha_val + float(rng.normal(scale=2.0)) if fit_alpha else float("-inf")
            x0 = _pack_params(alpha_j, rho_j, P_j, B_rj, B_ij, phi_j, fit_alpha, fit_phi)

        x0 = _clip_to_bounds(x0, bounds)
        if use_grad:
            # With analytic scores each iteration costs O(1) evaluations and
            # the convergence test can be much tighter than the ~0.4-unit
            # finite-difference noise floor allows (factr=1e4: relative ftol
            # ~2e-12, absolute ~4e-4 at lnL ~ 1e8).
            y0 = x0 / s_vec
            bounds_y = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, s_vec)]
            res = minimize(objective_with_grad_scaled, y0, method="L-BFGS-B",
                           jac=True, bounds=bounds_y,
                           options={"maxiter": max_iter,
                                    "maxfun": max(15000, 20 * max_iter),
                                    "ftol": 1e4 * np.finfo(float).eps})
            res.x = s_vec * res.x
        else:
            # maxfun must accommodate finite-difference gradients: each L-BFGS-B
            # iteration costs ~(dim+1) evaluations; scipy's default maxfun=15000
            # silently truncates a 240-dim fit after ~60 gradient evaluations.
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": max_iter, "eps": eps_vec,
                                    "maxfun": max(15000, 2 * max_iter * (len(x0) + 1))})
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res

    assert best_res is not None
    alpha_val_i, rho_c, P_c, B_r, B_i, phi_c = _unpack_params(best_res.x, n_coeff, r, fit_alpha, fit_phi)
    lnL = -float(best_fun)

    params = {
        "alpha_val": alpha_val_i,
        "rho_coeffs": rho_c,
        "P_coeffs": P_c,
        "B_real_coeffs": B_r,
        "B_imag_coeffs": B_i,
        "phi_coeffs": phi_c,
    }
    return params, lnL


def fit_nested_pair(
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    P_floor: np.ndarray,
    B_clip: np.ndarray,
    n_coeff: int,
    r: int,
    L: float,
    c0: float,
    lP: float,
    T_over_f2_avg: np.ndarray,
    T_over_fgamma_avg: np.ndarray | None = None,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    fit_phi: bool = True,
    max_iter: int = 500,
    n_starts: int = 2,
    seed: int | None = None,
    use_grad: bool = True,
) -> tuple[dict, float, dict, float]:
    """
    Fit the nested pair (H_env, H_env+signal) with symmetric optimization effort.

    The null is refit warm-started from the alternative's nuisance parameters
    (signal stripped); otherwise the extra starts used for the alternative can
    find a better optimum of the SHARED nuisance model and inflate the LR.
    Returns (env_params, lnL_env, qif_params, lnL_qif).
    """
    common = dict(
        L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg,
        T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
        use_planck_scale=use_planck_scale, fit_phi=fit_phi,
        max_iter=max_iter, n_starts=n_starts, use_grad=use_grad,
    )
    env_params, lnL_env = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        fit_alpha=False, seed=seed, **common,
    )
    qif_params, lnL_qif = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        fit_alpha=True, seed=seed, init_params=env_params, **common,
    )

    def _env_at(params: dict) -> float:
        # Direct evaluation of the NULL likelihood at a nuisance solution
        # (no fit). Whenever the alternative's optimizer finds a better basin
        # of the SHARED nuisance model, this hands the same basin to the null
        # for free and caps the LR at 2x the genuine alpha contribution.
        return float(loglike_et_qif(
            float("-inf"), params["rho_coeffs"], params["P_coeffs"],
            params["B_real_coeffs"], params["B_imag_coeffs"], params["phi_coeffs"],
            S_hat, m_eff, freqs, f_min, f_max, knots,
            L=L, c0=c0, lP=lP, gamma=gamma, use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg,
            P_floor=P_floor, B_clip=B_clip,
        ))

    # Iterated cross-pollination: a single ping-pong round is not enough --
    # the alternative refit can jump to a better nuisance basin AFTER the null
    # got its one chance, leaving a spurious LR > 0 with alpha at the floor.
    tol = 1e-6
    for _ in range(4):
        improved = False
        v = _env_at(qif_params)
        if v > lnL_env + tol:
            env_params = {k: (np.asarray(qif_params[k]).copy() if k != "alpha_val" else float("-inf"))
                          for k in qif_params}
            env_params["alpha_val"] = float("-inf")
            lnL_env = v
            improved = True
        env_params_x, lnL_env_x = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
            fit_alpha=False, seed=seed, init_params=env_params, **common,
        )
        if lnL_env_x > lnL_env + tol:
            env_params, lnL_env = env_params_x, lnL_env_x
            improved = True
        qif_params_x, lnL_qif_x = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
            fit_alpha=True, seed=seed, init_params=env_params, **common,
        )
        if lnL_qif_x > lnL_qif + tol:
            qif_params, lnL_qif = qif_params_x, lnL_qif_x
            improved = True
        if not improved:
            break
    # Final exact guard: the null gets the alternative's nuisances directly.
    v = _env_at(qif_params)
    if v > lnL_env:
        env_params = {k: (np.asarray(qif_params[k]).copy() if k != "alpha_val" else float("-inf"))
                      for k in qif_params}
        env_params["alpha_val"] = float("-inf")
        lnL_env = v
    return env_params, lnL_env, qif_params, lnL_qif


def _build_sigma_stack(
    params: dict,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    P_floor: np.ndarray,
    B_clip: np.ndarray,
    L: float,
    c0: float,
    lP: float,
    gamma: float,
    use_planck_scale: bool,
    T_over_f2_avg: np.ndarray,
    T_over_fgamma_avg: np.ndarray | None = None,
) -> np.ndarray:
    P_log = evaluate_spline(params["P_coeffs"], knots, freqs)
    P = np.exp(np.clip(P_log, -700.0, 700.0))
    P = np.maximum(P, P_floor)

    rho_logit = evaluate_spline(params["rho_coeffs"], knots, freqs).reshape(-1)
    rho = 1.0 / (1.0 + np.exp(-np.clip(rho_logit, -50.0, 50.0)))
    rho = np.clip(rho, 1e-6, 1.0 - 1e-6)

    B_real = evaluate_spline(params["B_real_coeffs"], knots, freqs)
    B_imag = evaluate_spline(params["B_imag_coeffs"], knots, freqs)
    phi_23 = evaluate_spline(params["phi_coeffs"], knots, freqs)
    phi_23 = wrap_phase(phi_23)

    if np.isneginf(params["alpha_val"]):
        alpha = 0.0
    else:
        alpha = float(np.exp(np.clip(params["alpha_val"], -700.0, 700.0)))

    if alpha > 0:
        S_path = qif_path_psd_binavg(
            alpha,
            c0,
            lP,
            L,
            f_min,
            f_max,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
        )
    else:
        S_path = np.zeros_like(freqs, dtype=float)

    Sigma = np.zeros((len(freqs), 3, 3), dtype=complex)
    for k in range(len(freqs)):
        Sigma_inst = np.diag(P[k, :]).astype(complex)
        Bk = (B_real[k] + 1j * B_imag[k]).astype(complex)
        if B_clip is not None:
            diag_B = np.sum(np.abs(Bk) ** 2, axis=1)
            dmax = float(np.max(diag_B))
            limit = float(B_clip[k])
            if np.isfinite(dmax) and np.isfinite(limit) and (dmax > limit) and (dmax > 0.0) and (limit > 0.0):
                Bk *= np.sqrt(limit / dmax)
        Sigma_env = Bk @ Bk.conj().T

        if alpha > 0:
            r_val = float(rho[k])
            M = np.array([[2.0, -r_val, -r_val],
                          [-r_val, 2.0, -r_val],
                          [-r_val, -r_val, 2.0]], dtype=float)
            Sigma_qif = (S_path[k] * M).astype(complex)
        else:
            Sigma_qif = np.zeros((3, 3), dtype=complex)

        G = np.diag([1.0, np.exp(1j * phi_23[k, 0]), np.exp(1j * phi_23[k, 1])]).astype(complex)
        Sigma_k = G @ (Sigma_inst + Sigma_env + Sigma_qif) @ G.conj().T
        Sigma[k] = hermitize(Sigma_k)
    return Sigma


def bootstrap_lr(
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    P_floor: np.ndarray,
    B_clip: np.ndarray,
    n_coeff: int,
    r: int,
    L: float,
    c0: float,
    lP: float,
    use_planck_scale: bool,
    T_over_f2_avg: np.ndarray,
    fit_phi: bool,
    T_over_fgamma_avg: np.ndarray | None = None,
    gamma: float = 2.0,
    n_boot: int = 10,
    refit: bool = True,
    seed: int = 0,
    max_iter: int = 200,
    n_starts: int = 3,
    bootstrap_mode: str = "round",
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)

    env_params, lnL_env, qif_params, lnL_qif = fit_nested_pair(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
        use_planck_scale=use_planck_scale,
        fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed,
    )
    lr_obs = 2.0 * (lnL_qif - lnL_env)

    Sigma_env = _build_sigma_stack(
        env_params,
        freqs,
        f_min,
        f_max,
        knots,
        P_floor,
        B_clip,
        L,
        c0,
        lP,
        gamma,
        use_planck_scale,
        T_over_f2_avg,
        T_over_fgamma_avg,
    )

    lrs = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        S_hat_b = np.zeros_like(S_hat)
        for k in range(len(freqs)):
            m_eff_k = float(m_eff[k])
            if bootstrap_mode == "probabilistic":
                m_star = draw_m_star(m_eff_k, rng)
            elif bootstrap_mode == "floor":
                m_star = int(np.floor(m_eff_k))
            elif bootstrap_mode == "ceil":
                m_star = int(np.ceil(m_eff_k))
            else:
                m_star = int(np.round(m_eff_k))
            m_star = max(1, m_star)
            S_hat_b[k] = sample_covariance_from_sigma(Sigma_env[k], m_star, rng)

        if refit:
            env_b, lnL_env_b, qif_b, lnL_qif_b = fit_nested_pair(
                S_hat_b, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
                L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
                use_planck_scale=use_planck_scale,
                fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed + i + 1,
            )
        else:
            lnL_env_b = loglike_et_qif(
                env_params["alpha_val"],
                env_params["rho_coeffs"],
                env_params["P_coeffs"],
                env_params["B_real_coeffs"],
                env_params["B_imag_coeffs"],
                env_params["phi_coeffs"],
                S_hat_b,
                m_eff,
                freqs,
                f_min,
                f_max,
                knots,
                L=L,
                c0=c0,
                lP=lP,
                gamma=gamma,
                use_planck_scale=use_planck_scale,
                T_over_f2_avg=T_over_f2_avg,
                T_over_fgamma_avg=T_over_fgamma_avg,
                P_floor=P_floor,
                B_clip=B_clip,
            )
            lnL_qif_b = loglike_et_qif(
                qif_params["alpha_val"],
                qif_params["rho_coeffs"],
                qif_params["P_coeffs"],
                qif_params["B_real_coeffs"],
                qif_params["B_imag_coeffs"],
                qif_params["phi_coeffs"],
                S_hat_b,
                m_eff,
                freqs,
                f_min,
                f_max,
                knots,
                L=L,
                c0=c0,
                lP=lP,
                gamma=gamma,
                use_planck_scale=use_planck_scale,
                T_over_f2_avg=T_over_f2_avg,
                T_over_fgamma_avg=T_over_fgamma_avg,
                P_floor=P_floor,
                B_clip=B_clip,
            )
        lrs[i] = 2.0 * (lnL_qif_b - lnL_env_b)

    pval = (np.sum(lrs >= lr_obs) + 1) / (n_boot + 1)
    se = np.sqrt(pval * (1.0 - pval) / max(1, n_boot))
    return lr_obs, pval, lrs


def _run_loglike_on_group(
    group: dict,
    max_seconds: float,
    nperseg_seconds: float,
    overlap: float,
    n_coeff: int,
    r: int,
    fmin_hz: float,
    fmax_hz: float | None,
    max_bins: int | None,
    fit: bool,
    fit_phi: bool,
    max_iter: int,
    bootstrap_n: int,
    bootstrap_refit: bool,
    stress_rank2: bool,
    calib_variants: bool,
    L: float,
    c0: float,
    lP: float,
    gamma: float,
    use_planck_scale: bool,
    transfer_data: tuple[np.ndarray, np.ndarray] | None,
    line_mask: list[tuple[float, float]] | None,
    n_starts: int,
    bootstrap_mode: str,
) -> dict:
    data, fs = _read_gwf_triplet(group["paths"], max_seconds=max_seconds)
    freqs, f_min, f_max, S_hat, m_eff = _compute_welch_csd_matrix(
        data,
        fs,
        nperseg_seconds=nperseg_seconds,
        overlap=overlap,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
        max_bins=max_bins,
        line_mask=line_mask,
    )

    if len(freqs) < n_coeff:
        raise ValueError("Not enough frequency bins for the requested spline coefficients.")

    knots = _open_log_knots(freqs[0], freqs[-1], n_coeff, k=3)
    rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs = _build_initial_coeffs(
        S_hat, n_coeff, r, freqs=freqs, knots=knots)

    valid = np.all(np.isfinite(np.real(np.diagonal(S_hat, axis1=1, axis2=2))), axis=1)
    P_floor, B_clip = compute_fixed_thresholds(S_hat, valid)

    T_over_f2_avg = None
    T_over_fgamma_avg = None
    if transfer_data is not None:
        f_src, t_sq_src = transfer_data
        if np.isclose(gamma, 2.0):
            T_over_f2_avg = compute_T_over_f2_avg_simpson10(
                lambda f: _transfer_sq_func(f, f_src, t_sq_src),
                f_min,
                f_max,
            )
        else:
            T_over_fgamma_avg = compute_T_over_fgamma_avg_simpson10(
                lambda f: _transfer_sq_func(f, f_src, t_sq_src),
                f_min,
                f_max,
                gamma,
            )
    else:
        if np.isclose(gamma, 2.0):
            T_over_f2_avg = 1.0 / np.maximum(f_min * f_max, 1e-300)
        else:
            # assume |T|^2≈1 in-band; use analytic bin-average of f^-gamma
            if np.isclose(gamma, 1.0):
                avg = (
                    np.log(np.maximum(f_max, 1e-300)) - np.log(np.maximum(f_min, 1e-300))
                ) / np.maximum(f_max - f_min, 1e-300)
            else:
                avg = (
                    np.maximum(f_max, 1e-300) ** (1.0 - gamma)
                    - np.maximum(f_min, 1e-300) ** (1.0 - gamma)
                ) / np.maximum(1.0 - gamma, 1e-9)
                avg = avg / np.maximum(f_max - f_min, 1e-300)
            T_over_fgamma_avg = avg

    results: dict = {
        "group": group,
        "freq_bins": len(freqs),
    }

    if not fit:
        # Evaluate the NULL model (alpha = 0) at the initial coefficients; using
        # alpha = 1 here poisons Sigma with a signal ~40 orders above the data
        # and makes the result data-independent.
        lnL = loglike_et_qif(
            float("-inf"),
            rho_coeffs,
            P_coeffs,
            B_real_coeffs,
            B_imag_coeffs,
            phi_coeffs,
            S_hat,
            m_eff,
            freqs,
            f_min,
            f_max,
            knots,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            P_floor=P_floor,
            B_clip=B_clip,
        )
        results["loglike"] = float(lnL)
        return results

    env_params, lnL_env, qif_params, lnL_qif = fit_nested_pair(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L,
        c0=c0,
        lP=lP,
        use_planck_scale=use_planck_scale,
        T_over_f2_avg=T_over_f2_avg,
        T_over_fgamma_avg=T_over_fgamma_avg,
        gamma=gamma,
        fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"],
    )
    lr = 2.0 * (lnL_qif - lnL_env)
    A_h_hat = float(np.exp(qif_params["alpha_val"])) if np.isfinite(qif_params["alpha_val"]) else 0.0
    results["baseline"] = {
        "r": r,
        "fit_phi": fit_phi,
        "lnL_env": float(lnL_env),
        "lnL_qif": float(lnL_qif),
        "lr": float(lr),
        "A_h_hat": A_h_hat,
    }

    if bootstrap_n > 0:
        lr_obs, pval, lrs = bootstrap_lr(
            S_hat,
            m_eff,
            freqs,
            f_min,
            f_max,
            knots,
            P_floor,
            B_clip,
            n_coeff,
            r,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_phi=fit_phi,
            n_boot=bootstrap_n,
            refit=bootstrap_refit,
            seed=group["gps"] % 100000,
            max_iter=max_iter,
            n_starts=n_starts,
            bootstrap_mode=bootstrap_mode,
        )
        results["baseline"]["lr_bootstrap"] = float(lr_obs)
        results["baseline"]["p_value"] = float(pval)
        results["baseline"]["p_se"] = float(np.sqrt(pval * (1.0 - pval) / max(1, bootstrap_n)))

    if stress_rank2:
        env_params2, lnL_env2, qif_params2, lnL_qif2 = fit_nested_pair(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, 2,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 2,
        )
        lr2 = 2.0 * (lnL_qif2 - lnL_env2)
        results["stress_rank2"] = {
            "r": 2,
            "fit_phi": fit_phi,
            "lnL_env": float(lnL_env2),
            "lnL_qif": float(lnL_qif2),
            "lr": float(lr2),
        }

    if calib_variants:
        env_params0, lnL_env0, qif_params0, lnL_qif0 = fit_nested_pair(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_phi=False, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 3,
        )
        lr0 = 2.0 * (lnL_qif0 - lnL_env0)
        results["calib_phi_fixed"] = {
            "r": r,
            "fit_phi": False,
            "lnL_env": float(lnL_env0),
            "lnL_qif": float(lnL_qif0),
            "lr": float(lr0),
        }

    return results


def run_on_sample_data(
    data_root: str,
    max_seconds: float = 256.0,
    nperseg_seconds: float = 4.0,
    overlap: float = 0.5,
    n_coeff: int = 20,
    r: int = 1,
    fmin_hz: float = 10.0,
    fmax_hz: float | None = None,
    max_bins: int | None = None,
    fit: bool = False,
    fit_phi: bool = True,
    max_iter: int = 200,
    bootstrap_n: int = 0,
    bootstrap_refit: bool = True,
    stress_rank2: bool = False,
    calib_variants: bool = False,
    L: float = 1.0e4,
    c0: float = 299792458.0,
    lP: float = 1.616255e-35,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    transfer_data: tuple[np.ndarray, np.ndarray] | None = None,
    line_mask: list[tuple[float, float]] | None = None,
    n_starts: int = 3,
    bootstrap_mode: str = "round",
) -> None:
    groups = _find_gwf_groups(data_root)
    if not groups:
        print(f"No .gwf files found under {data_root}.")
        return
    for group in groups:
        results = _run_loglike_on_group(
            group,
            max_seconds=max_seconds,
            nperseg_seconds=nperseg_seconds,
            overlap=overlap,
            n_coeff=n_coeff,
            r=r,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            max_bins=max_bins,
            fit=fit,
            fit_phi=fit_phi,
            max_iter=max_iter,
            bootstrap_n=bootstrap_n,
            bootstrap_refit=bootstrap_refit,
            stress_rank2=stress_rank2,
            calib_variants=calib_variants,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            transfer_data=transfer_data,
            line_mask=line_mask,
            n_starts=n_starts,
            bootstrap_mode=bootstrap_mode,
        )
        label = group["set"] or "data"
        if "loglike" in results:
            print(f"{label} gps={group['gps']} loglike_et_qif={results['loglike']:.6e}")
            continue
        base = results["baseline"]
        line = f"{label} gps={group['gps']} bins={results['freq_bins']} r={base['r']} lr={base['lr']:.6e} A_h_hat={base.get('A_h_hat', float('nan')):.3e}"
        if "p_value" in base:
            line += f" p={base['p_value']:.4f}±{base['p_se']:.4f}"
        print(line)
        if "stress_rank2" in results:
            r2 = results["stress_rank2"]
            print(f"  rank2 stress: lr={r2['lr']:.6e}")
        if "calib_phi_fixed" in results:
            cv = results["calib_phi_fixed"]
            print(f"  phi_fixed: lr={cv['lr']:.6e}")


def run_synthetic() -> None:
    data = _make_synthetic_data()
    lnL = loglike_et_qif(
        data["alpha_val"],
        data["rho_coeffs"],
        data["P_coeffs"],
        data["B_real_coeffs"],
        data["B_imag_coeffs"],
        data["phi_coeffs"],
        data["S_hat"],
        data["m_eff"],
        data["freqs"],
        data["f_min"],
        data["f_max"],
        data["knots"],
        data["L"],
        data["c0"],
        data["lP"],
        use_planck_scale=False,
        T_over_f2_avg=data["T_over_f2_avg"],
        P_floor=data["P_floor"],
        B_clip=data["B_clip"],
    )
    print(f"synthetic loglike_et_qif={lnL:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QIF likelihood on ET MDC GWF samples.")
    parser.add_argument("--data-root", default="data", help="Root directory containing GWF files.")
    parser.add_argument("--max-seconds", type=float, default=256.0, help="Seconds to read from each GWF.")
    parser.add_argument("--nperseg-seconds", type=float, default=4.0, help="Welch segment length in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Welch overlap fraction (0-1).")
    parser.add_argument("--n-coeff", type=int, default=20, help="Number of spline coefficients (must be rich enough to represent the PSD shape; too few coefficients lets the f^-gamma term absorb spline misfit).")
    parser.add_argument("--r", type=int, default=1, help="Environmental rank r.")
    parser.add_argument("--fmin", type=float, default=10.0, help="Minimum frequency (Hz).")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency (Hz).")
    parser.add_argument("--max-bins", type=int, default=None, help="Optional max frequency bins (downsample).")
    parser.add_argument("--fit", action="store_true", help="Fit spline coefficients (optimize log-likelihood).")
    parser.add_argument("--fit-phi", action="store_true", help="Fit calibration phase coefficients.")
    parser.add_argument("--max-iter", type=int, default=500, help="Max optimizer iterations.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap replicates for p-value.")
    parser.add_argument("--bootstrap-refit", action="store_true", help="Refit models inside bootstrap.")
    parser.add_argument("--stress-rank2", action="store_true", help="Run rank-2 environmental stress test.")
    parser.add_argument("--calib-variants", action="store_true", help="Run calibration phi fixed vs fitted variant.")
    parser.add_argument("--n-starts", type=int, default=3, help="Optimizer multi-start count.")
    parser.add_argument("--L", type=float, default=1.0e4, help="Effective path length (m).")
    parser.add_argument("--c0", type=float, default=299792458.0, help="Speed of light (m/s).")
    parser.add_argument("--lP", type=float, default=1.616255e-35, help="Planck length (m).")
    parser.add_argument("--gamma", type=float, default=2.0, help="Spectral index for f^-gamma.")
    parser.add_argument("--use-planck-scaling", action="store_true",
                        help="Interpret alpha as dimensionless and map to A_h via (alpha*c0*lP)/(2*pi^2*L^2).")
    parser.add_argument("--transfer-csv", type=str, default=None, help="CSV with transfer function (f, T or f, Re, Im).")
    parser.add_argument("--transfer-sq", action="store_true", help="Interpret transfer CSV second column as |T|^2.")
    parser.add_argument("--line-mask", type=str, default=None, help="CSV or whitespace file with fmin,fmax per line.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic demo only.")
    parser.add_argument("--bootstrap-mode", type=str, default="round",
                        choices=["round", "probabilistic", "floor", "ceil"],
                        help="How to convert fractional m_eff to integer m* in bootstrap.")
    args = parser.parse_args()

    if args.synthetic:
        run_synthetic()
        return

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_root))
    if not os.path.isdir(base):
        print(f"Data root not found: {base}. Running synthetic demo.")
        run_synthetic()
        return

    line_mask = _load_line_mask(args.line_mask) if args.line_mask else None
    transfer_data = None
    if args.transfer_csv:
        transfer_data = _load_transfer_csv(args.transfer_csv, is_sq=args.transfer_sq)

    run_on_sample_data(
        base,
        max_seconds=args.max_seconds,
        nperseg_seconds=args.nperseg_seconds,
        overlap=args.overlap,
        n_coeff=args.n_coeff,
        r=args.r,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        max_bins=args.max_bins,
        fit=args.fit,
        fit_phi=args.fit_phi,
        max_iter=args.max_iter,
        bootstrap_n=args.bootstrap,
        bootstrap_refit=args.bootstrap_refit,
        stress_rank2=args.stress_rank2,
        calib_variants=args.calib_variants,
        L=args.L,
        c0=args.c0,
        lP=args.lP,
        gamma=args.gamma,
        use_planck_scale=args.use_planck_scaling,
        transfer_data=transfer_data,
        line_mask=line_mask,
        n_starts=args.n_starts,
        bootstrap_mode=args.bootstrap_mode,
    )


if __name__ == "__main__":
    main()
