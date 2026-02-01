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
    clip_logP: float = 50.0,
    clip_logit_rho: float = 50.0,
    clip_log_alpha: float = 100.0,
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

    lnL = 0.0
    eye3 = np.eye(3, dtype=complex)

    # elementwise clip for env factors (overflow guard), then safe rescaling clip
    if B_clip is not None:
        if B_clip.shape != (Nf,):
            raise ValueError("B_clip must have shape (N_f,)")
        B_elem_clip = B_elem_clip_factor * np.sqrt(np.maximum(B_clip, jitter_floor))
        B_elem_clip = B_elem_clip.reshape(Nf, 1, 1)
        B_real = np.clip(B_real, -B_elem_clip, B_elem_clip)
        B_imag = np.clip(B_imag, -B_elem_clip, B_elem_clip)

    for k in range(Nf):
        if m_eff[k] <= 0:
            continue
        Sk = S_hat[k]
        if not np.all(np.isfinite(Sk)):
            continue

        # instrument
        Sigma_inst = np.diag(P[k, :]).astype(complex)

        # environment (rank r)
        Bk = (B_real[k] + 1j * B_imag[k]).astype(complex)  # (3, r)
        if B_clip is not None:
            diag_B = np.sum(np.abs(Bk) ** 2, axis=1)  # diag(BB^H)
            dmax = float(np.max(diag_B))
            limit = float(B_clip[k])
            if np.isfinite(dmax) and np.isfinite(limit) and (dmax > limit) and (dmax > 0.0) and (limit > 0.0):
                Bk *= np.sqrt(limit / dmax)
        Sigma_env = Bk @ Bk.conj().T

        # QIF
        if alpha > 0:
            r_val = float(rho[k])
            M = np.array([[2.0, -r_val, -r_val],
                          [-r_val, 2.0, -r_val],
                          [-r_val, -r_val, 2.0]], dtype=float)
            Sigma_qif = (S_path[k] * M).astype(complex)
        else:
            Sigma_qif = np.zeros((3, 3), dtype=complex)

        # calibration (phase-only, gauge-fixed)
        G = np.diag([1.0, np.exp(1j * phi_23[k, 0]), np.exp(1j * phi_23[k, 1])]).astype(complex)

        Sigma = G @ (Sigma_inst + Sigma_env + Sigma_qif) @ G.conj().T

        # hermitize model and data
        Sigma = hermitize(Sigma)
        Sk = hermitize(Sk)

        # jitter hardening
        max_diag = float(np.max(np.real(np.diag(Sigma))))
        Sigma = Sigma + (eps * max_diag + jitter_floor) * eye3

        # Wishart terms via Cholesky
        try:
            chol, lower = scipy.linalg.cho_factor(Sigma, lower=True, check_finite=False)
        except scipy.linalg.LinAlgError:
            return -np.inf

        logdet = 2.0 * np.sum(np.log(np.maximum(np.real(np.diag(chol)), 1e-300)))
        Sigma_inv_S = scipy.linalg.cho_solve((chol, lower), Sk, check_finite=False)
        tr = float(np.trace(Sigma_inv_S).real)

        lnL -= float(m_eff[k]) * (float(logdet) + tr)

    return float(lnL)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    if max_bins is not None and len(freqs) > max_bins:
        idx = np.linspace(0, len(freqs) - 1, max_bins).round().astype(int)
        freqs = freqs[idx]
        S_hat = S_hat[idx]

    m_eff_val = _effective_m(nseg)
    m_eff = np.full_like(freqs, float(m_eff_val))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    f_min = freqs - 0.5 * df
    f_max = freqs + 0.5 * df

    valid = f_min > 0
    return freqs[valid], f_min[valid], f_max[valid], S_hat[valid], m_eff[valid]


def _build_initial_coeffs(S_hat: np.ndarray, n_coeff: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag = np.real(np.diagonal(S_hat, axis1=1, axis2=2))
    median_diag = np.maximum(np.median(diag, axis=0), 1e-300)
    P_coeffs = np.tile(np.log(median_diag), (n_coeff, 1))
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


def _default_bounds(n_coeff: int, r: int, fit_alpha: bool, fit_phi: bool) -> list[tuple[float, float] | None]:
    bounds: list[tuple[float, float] | None] = []
    if fit_alpha:
        bounds.append((ALPHA_LOG_MIN, ALPHA_LOG_MAX))
    bounds.extend([(-10.0, 10.0)] * n_coeff)           # rho logits
    bounds.extend([(-30.0, 30.0)] * (n_coeff * 3))     # log P
    bounds.extend([(-1e3, 1e3)] * (n_coeff * 3 * r))    # B real
    bounds.extend([(-1e3, 1e3)] * (n_coeff * 3 * r))    # B imag
    if fit_phi:
        bounds.extend([(-np.pi, np.pi)] * (n_coeff * 2))
    return bounds


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
    alpha_init: float | None = None,
) -> tuple[dict, float]:
    rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs = _build_initial_coeffs(S_hat, n_coeff, r)
    if init_params is not None:
        rho_coeffs = init_params.get("rho_coeffs", rho_coeffs)
        P_coeffs = init_params.get("P_coeffs", P_coeffs)
        B_real_coeffs = init_params.get("B_real_coeffs", B_real_coeffs)
        B_imag_coeffs = init_params.get("B_imag_coeffs", B_imag_coeffs)
        phi_coeffs = init_params.get("phi_coeffs", phi_coeffs)

    alpha_val = float(np.log(1.0))
    if alpha_init is not None:
        alpha_val = float(alpha_init)
    if fit_alpha:
        alpha_val = float(np.clip(alpha_val, ALPHA_LOG_MIN, ALPHA_LOG_MAX))

    bounds = _default_bounds(n_coeff, r, fit_alpha, fit_phi)

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

    rng = np.random.default_rng(seed)
    best_res = None
    best_fun = np.inf
    for start in range(max(1, n_starts)):
        if start == 0:
            x0 = _pack_params(alpha_val, rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs, fit_alpha, fit_phi)
        else:
            rho_j = rho_coeffs + rng.normal(scale=0.1, size=rho_coeffs.shape)
            P_j = P_coeffs + rng.normal(scale=0.5, size=P_coeffs.shape)
            B_rj = B_real_coeffs + rng.normal(scale=1e-3, size=B_real_coeffs.shape)
            B_ij = B_imag_coeffs + rng.normal(scale=1e-3, size=B_imag_coeffs.shape)
            phi_j = phi_coeffs + rng.normal(scale=0.1, size=phi_coeffs.shape)
            alpha_j = alpha_val + float(rng.normal(scale=1.0)) if fit_alpha else float("-inf")
            x0 = _pack_params(alpha_j, rho_j, P_j, B_rj, B_ij, phi_j, fit_alpha, fit_phi)

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
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
    P = np.exp(np.clip(P_log, -50.0, 50.0))
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
        alpha = float(np.exp(np.clip(params["alpha_val"], -100.0, 100.0)))

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

    env_params, lnL_env = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
        use_planck_scale=use_planck_scale,
        fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed
    )
    qif_params, lnL_qif = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
        use_planck_scale=use_planck_scale,
        fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed,
        init_params=env_params, alpha_init=ALPHA_LOG_MIN
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
            env_b, lnL_env_b = fit_model(
                S_hat_b, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
                L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
                use_planck_scale=use_planck_scale,
                fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed + i + 1
            )
            qif_b, lnL_qif_b = fit_model(
                S_hat_b, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
                L=L, c0=c0, lP=lP, T_over_f2_avg=T_over_f2_avg, T_over_fgamma_avg=T_over_fgamma_avg, gamma=gamma,
                use_planck_scale=use_planck_scale,
                fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed + i + 1,
                init_params=env_b, alpha_init=ALPHA_LOG_MIN
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

    knots = _open_uniform_knots(freqs[0], freqs[-1], n_coeff, k=3)
    rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs = _build_initial_coeffs(S_hat, n_coeff, r)

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
        lnL = loglike_et_qif(
            np.log(1.0),
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

    env_params, lnL_env = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L,
        c0=c0,
        lP=lP,
        use_planck_scale=use_planck_scale,
        T_over_f2_avg=T_over_f2_avg,
        T_over_fgamma_avg=T_over_fgamma_avg,
        gamma=gamma,
        fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"]
    )
    qif_params, lnL_qif = fit_model(
        S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
        L=L,
        c0=c0,
        lP=lP,
        use_planck_scale=use_planck_scale,
        T_over_f2_avg=T_over_f2_avg,
        T_over_fgamma_avg=T_over_fgamma_avg,
        gamma=gamma,
        fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"],
        init_params=env_params, alpha_init=ALPHA_LOG_MIN
    )
    lr = 2.0 * (lnL_qif - lnL_env)
    results["baseline"] = {
        "r": r,
        "fit_phi": fit_phi,
        "lnL_env": float(lnL_env),
        "lnL_qif": float(lnL_qif),
        "lr": float(lr),
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
        env_params2, lnL_env2 = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, 2,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 2
        )
        qif_params2, lnL_qif2 = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, 2,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 2,
            init_params=env_params2, alpha_init=ALPHA_LOG_MIN
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
        env_params0, lnL_env0 = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_alpha=False, fit_phi=False, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 3
        )
        qif_params0, lnL_qif0 = fit_model(
            S_hat, m_eff, freqs, f_min, f_max, knots, P_floor, B_clip, n_coeff, r,
            L=L,
            c0=c0,
            lP=lP,
            use_planck_scale=use_planck_scale,
            T_over_f2_avg=T_over_f2_avg,
            T_over_fgamma_avg=T_over_fgamma_avg,
            gamma=gamma,
            fit_alpha=True, fit_phi=False, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 3,
            init_params=env_params0, alpha_init=ALPHA_LOG_MIN
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
    n_coeff: int = 6,
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
        line = f"{label} gps={group['gps']} bins={results['freq_bins']} r={base['r']} lr={base['lr']:.6e}"
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
    parser.add_argument("--n-coeff", type=int, default=6, help="Number of spline coefficients.")
    parser.add_argument("--r", type=int, default=1, help="Environmental rank r.")
    parser.add_argument("--fmin", type=float, default=10.0, help="Minimum frequency (Hz).")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency (Hz).")
    parser.add_argument("--max-bins", type=int, default=None, help="Optional max frequency bins (downsample).")
    parser.add_argument("--fit", action="store_true", help="Fit spline coefficients (optimize log-likelihood).")
    parser.add_argument("--fit-phi", action="store_true", help="Fit calibration phase coefficients.")
    parser.add_argument("--max-iter", type=int, default=200, help="Max optimizer iterations.")
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
