import argparse
import os
from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from qif_v2 import (
    _build_initial_coeffs,
    _compute_welch_csd_matrix,
    _default_bounds,
    _find_gwf_groups,
    _load_line_mask,
    _load_transfer_csv,
    _make_synthetic_data,
    _open_uniform_knots,
    _pack_params,
    _read_gwf_triplet,
    _transfer_sq_func,
    _unpack_params,
    compute_fixed_thresholds,
    compute_T_over_f2_avg_simpson10,
    compute_T_over_fgamma_avg_simpson10,
    draw_m_star,
)


class Backend:
    def __init__(self, xp, name: str, is_cuda: bool) -> None:
        self.xp = xp
        self.name = name
        self.is_cuda = is_cuda

    def asarray(self, arr, dtype=None):
        if self.is_cuda:
            return self.xp.asarray(arr, dtype=dtype)
        if dtype is None:
            return np.asarray(arr)
        return np.asarray(arr, dtype=dtype)

    def asnumpy(self, arr):
        if self.is_cuda:
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


def _try_import_cupy():
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        _ = cp.zeros((1,), dtype=cp.float32)
        if hasattr(cp.cuda.runtime, "getDeviceCount"):
            if cp.cuda.runtime.getDeviceCount() <= 0:
                return None
    except Exception:
        return None
    return cp


def get_backend(device: str = "auto") -> Backend:
    device = device.lower()
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu")
    if device == "cpu":
        return Backend(np, "numpy", False)

    cp = _try_import_cupy()
    if cp is None:
        if device == "gpu":
            raise RuntimeError("CUDA backend requested but CuPy is not available.")
        return Backend(np, "numpy", False)
    return Backend(cp, "cupy", True)


def hermitize_xp(A, xp):
    return 0.5 * (A + A.conj().T)


def wrap_phase_xp(phi, xp):
    return (phi + xp.pi) % (2.0 * xp.pi) - xp.pi


def evaluate_spline_cpu(coeffs: np.ndarray, knots: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    spline = BSpline(knots, coeffs, k=3, extrapolate=False)
    return spline(freqs)


def _eval_spline_xp(coeffs, knots, freqs_cpu, backend: Backend):
    out = evaluate_spline_cpu(coeffs, knots, freqs_cpu)
    return backend.asarray(out)


def qif_path_psd_binavg_xp(
    alpha: float,
    c0: float,
    lP: float,
    L: float,
    f_min,
    f_max,
    xp,
    T_eff_sq=None,
    T_over_f2_avg=None,
    T_over_fgamma_avg=None,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
):
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
        T_eff_sq = xp.ones_like(f_min, dtype=float)

    if np.isclose(gamma, 1.0):
        denom = xp.maximum(xp.log(xp.maximum(f_max, 1e-300)) - xp.log(xp.maximum(f_min, 1e-300)), 1e-300)
        avg = denom / xp.maximum(f_max - f_min, 1e-300)
    else:
        denom = xp.maximum(1.0 - gamma, 1e-9)
        avg = (xp.maximum(f_max, 1e-300) ** (1.0 - gamma) - xp.maximum(f_min, 1e-300) ** (1.0 - gamma)) / denom
        avg = avg / xp.maximum(f_max - f_min, 1e-300)
    return A * T_eff_sq * avg


@dataclass
class QIFData:
    freqs: np.ndarray
    knots: np.ndarray
    S_hat_x: object
    S_hat_cpu: np.ndarray
    m_eff: np.ndarray
    f_min_x: object
    f_max_x: object
    P_floor_x: object | None
    B_clip_x: object | None
    T_over_f2_avg_x: object | None
    T_over_fgamma_avg_x: object | None


def loglike_et_qif_cuda(
    alpha_val: float,
    rho_coeffs: np.ndarray,
    P_coeffs: np.ndarray,
    B_real_coeffs: np.ndarray,
    B_imag_coeffs: np.ndarray,
    phi_coeffs: np.ndarray,
    data: QIFData,
    L: float,
    c0: float,
    lP: float,
    gamma: float = 2.0,
    use_planck_scale: bool = False,
    T_eff_sq=None,
    eps: float = 1e-9,
    jitter_floor: float = 1e-50,
    clip_logP: float = 50.0,
    clip_logit_rho: float = 50.0,
    clip_log_alpha: float = 100.0,
    B_elem_clip_factor: float = 1e6,
    backend: Backend | None = None,
) -> float:
    if backend is None:
        backend = Backend(np, "numpy", False)
    xp = backend.xp

    S_hat = data.S_hat_x
    m_eff = data.m_eff
    freqs_cpu = data.freqs
    f_min = data.f_min_x
    f_max = data.f_max_x

    if S_hat.ndim != 3 or S_hat.shape[1:] != (3, 3):
        raise ValueError("S_hat must have shape (N_f, 3, 3).")
    Nf = int(S_hat.shape[0])
    if m_eff.shape != (Nf,):
        raise ValueError("m_eff must have shape (N_f,)")

    P_log_cpu = evaluate_spline_cpu(P_coeffs, data.knots, freqs_cpu)
    if np.any(~np.isfinite(P_log_cpu)):
        return -np.inf
    P = xp.exp(xp.clip(backend.asarray(P_log_cpu), -clip_logP, clip_logP))
    if data.P_floor_x is not None:
        P = xp.maximum(P, data.P_floor_x)
    else:
        P = xp.maximum(P, jitter_floor)

    rho_logit_cpu = evaluate_spline_cpu(rho_coeffs, data.knots, freqs_cpu).reshape(-1)
    if np.any(~np.isfinite(rho_logit_cpu)):
        return -np.inf
    rho = 1.0 / (1.0 + xp.exp(-xp.clip(backend.asarray(rho_logit_cpu), -clip_logit_rho, clip_logit_rho)))
    rho = xp.clip(rho, 1e-6, 1.0 - 1e-6)

    B_real_cpu = evaluate_spline_cpu(B_real_coeffs, data.knots, freqs_cpu)
    B_imag_cpu = evaluate_spline_cpu(B_imag_coeffs, data.knots, freqs_cpu)
    if np.any(~np.isfinite(B_real_cpu)) or np.any(~np.isfinite(B_imag_cpu)):
        return -np.inf
    B_real = backend.asarray(B_real_cpu)
    B_imag = backend.asarray(B_imag_cpu)

    phi_23_cpu = evaluate_spline_cpu(phi_coeffs, data.knots, freqs_cpu)
    if np.any(~np.isfinite(phi_23_cpu)):
        return -np.inf
    phi_23 = wrap_phase_xp(backend.asarray(phi_23_cpu), xp)

    if np.isneginf(alpha_val):
        alpha = 0.0
    else:
        alpha = float(np.exp(np.clip(alpha_val, -clip_log_alpha, clip_log_alpha)))

    if alpha > 0:
        S_path = qif_path_psd_binavg_xp(
            alpha,
            c0,
            lP,
            L,
            f_min,
            f_max,
            xp,
            T_eff_sq=T_eff_sq,
            T_over_f2_avg=data.T_over_f2_avg_x,
            T_over_fgamma_avg=data.T_over_fgamma_avg_x,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
        )
    else:
        S_path = xp.zeros((Nf,), dtype=float)

    lnL = 0.0
    eye3 = xp.eye(3, dtype=complex)

    if data.B_clip_x is not None:
        B_elem_clip = B_elem_clip_factor * xp.sqrt(xp.maximum(data.B_clip_x, jitter_floor))
        B_elem_clip = B_elem_clip.reshape(Nf, 1, 1)
        B_real = xp.clip(B_real, -B_elem_clip, B_elem_clip)
        B_imag = xp.clip(B_imag, -B_elem_clip, B_elem_clip)

    for k in range(Nf):
        if m_eff[k] <= 0:
            continue
        Sk = S_hat[k]
        if not bool(xp.all(xp.isfinite(Sk))):
            continue

        Sigma_inst = xp.diag(P[k, :]).astype(complex)
        Bk = (B_real[k] + 1j * B_imag[k]).astype(complex)
        if data.B_clip_x is not None:
            diag_B = xp.sum(xp.abs(Bk) ** 2, axis=1)
            dmax = float(xp.max(diag_B))
            limit = float(data.B_clip_x[k])
            if np.isfinite(dmax) and np.isfinite(limit) and (dmax > limit) and (dmax > 0.0) and (limit > 0.0):
                Bk *= np.sqrt(limit / dmax)
        Sigma_env = Bk @ Bk.conj().T

        if alpha > 0:
            r_val = float(rho[k])
            M = xp.array([[2.0, -r_val, -r_val],
                          [-r_val, 2.0, -r_val],
                          [-r_val, -r_val, 2.0]], dtype=float)
            Sigma_qif = (S_path[k] * M).astype(complex)
        else:
            Sigma_qif = xp.zeros((3, 3), dtype=complex)

        G = xp.diag([1.0, xp.exp(1j * phi_23[k, 0]), xp.exp(1j * phi_23[k, 1])]).astype(complex)
        Sigma = G @ (Sigma_inst + Sigma_env + Sigma_qif) @ G.conj().T

        Sigma = hermitize_xp(Sigma, xp)
        Sk = hermitize_xp(Sk, xp)

        max_diag = float(xp.max(xp.real(xp.diag(Sigma))))
        Sigma = Sigma + (eps * max_diag + jitter_floor) * eye3

        try:
            chol = xp.linalg.cholesky(Sigma)
        except Exception:
            return -np.inf

        logdet = 2.0 * xp.sum(xp.log(xp.maximum(xp.real(xp.diag(chol)), 1e-300)))
        Sigma_inv_S = xp.linalg.solve(Sigma, Sk)
        tr = xp.real(xp.trace(Sigma_inv_S))

        lnL -= float(m_eff[k]) * (float(logdet) + float(tr))

    return float(lnL)


def _build_sigma_stack_cuda(
    params: dict,
    data: QIFData,
    L: float,
    c0: float,
    lP: float,
    gamma: float,
    use_planck_scale: bool,
    backend: Backend,
) -> object:
    xp = backend.xp
    freqs_cpu = data.freqs

    P_log_cpu = evaluate_spline_cpu(params["P_coeffs"], data.knots, freqs_cpu)
    P = xp.exp(xp.clip(backend.asarray(P_log_cpu), -50.0, 50.0))
    if data.P_floor_x is not None:
        P = xp.maximum(P, data.P_floor_x)

    rho_logit_cpu = evaluate_spline_cpu(params["rho_coeffs"], data.knots, freqs_cpu).reshape(-1)
    rho = 1.0 / (1.0 + xp.exp(-xp.clip(backend.asarray(rho_logit_cpu), -50.0, 50.0)))
    rho = xp.clip(rho, 1e-6, 1.0 - 1e-6)

    B_real_cpu = evaluate_spline_cpu(params["B_real_coeffs"], data.knots, freqs_cpu)
    B_imag_cpu = evaluate_spline_cpu(params["B_imag_coeffs"], data.knots, freqs_cpu)
    B_real = backend.asarray(B_real_cpu)
    B_imag = backend.asarray(B_imag_cpu)

    phi_23_cpu = evaluate_spline_cpu(params["phi_coeffs"], data.knots, freqs_cpu)
    phi_23 = wrap_phase_xp(backend.asarray(phi_23_cpu), xp)

    if np.isneginf(params["alpha_val"]):
        alpha = 0.0
    else:
        alpha = float(np.exp(np.clip(params["alpha_val"], -100.0, 100.0)))

    if alpha > 0:
        S_path = qif_path_psd_binavg_xp(
            alpha,
            c0,
            lP,
            L,
            data.f_min_x,
            data.f_max_x,
            xp,
            T_over_f2_avg=data.T_over_f2_avg_x,
            T_over_fgamma_avg=data.T_over_fgamma_avg_x,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
        )
    else:
        S_path = xp.zeros((len(freqs_cpu),), dtype=float)

    Sigma = xp.zeros((len(freqs_cpu), 3, 3), dtype=complex)
    for k in range(len(freqs_cpu)):
        Sigma_inst = xp.diag(P[k, :]).astype(complex)
        Bk = (B_real[k] + 1j * B_imag[k]).astype(complex)
        if data.B_clip_x is not None:
            diag_B = xp.sum(xp.abs(Bk) ** 2, axis=1)
            dmax = float(xp.max(diag_B))
            limit = float(data.B_clip_x[k])
            if np.isfinite(dmax) and np.isfinite(limit) and (dmax > limit) and (dmax > 0.0) and (limit > 0.0):
                Bk *= np.sqrt(limit / dmax)
        Sigma_env = Bk @ Bk.conj().T

        if alpha > 0:
            r_val = float(rho[k])
            M = xp.array([[2.0, -r_val, -r_val],
                          [-r_val, 2.0, -r_val],
                          [-r_val, -r_val, 2.0]], dtype=float)
            Sigma_qif = (S_path[k] * M).astype(complex)
        else:
            Sigma_qif = xp.zeros((3, 3), dtype=complex)

        G = xp.diag([1.0, xp.exp(1j * phi_23[k, 0]), xp.exp(1j * phi_23[k, 1])]).astype(complex)
        Sigma_k = G @ (Sigma_inst + Sigma_env + Sigma_qif) @ G.conj().T
        Sigma[k] = hermitize_xp(Sigma_k, xp)
    return Sigma


def _make_rng(backend: Backend, seed: int):
    xp = backend.xp
    if backend.is_cuda:
        try:
            return xp.random.default_rng(seed)
        except AttributeError:
            return xp.random.RandomState(seed)
    return np.random.default_rng(seed)


def _standard_normal(rng, shape):
    try:
        return rng.standard_normal(shape)
    except AttributeError:
        return rng.randn(*shape)


def sample_covariance_from_sigma_xp(Sigma, m_star: int, rng, xp):
    if m_star <= 0:
        return xp.zeros_like(Sigma)
    L = xp.linalg.cholesky(Sigma)
    z = _standard_normal(rng, (Sigma.shape[0], m_star)) + 1j * _standard_normal(rng, (Sigma.shape[0], m_star))
    z = z / xp.sqrt(2.0)
    y = L @ z
    return (y @ y.conj().T) / float(m_star)


def fit_model_cuda(
    data: QIFData,
    n_coeff: int,
    r: int,
    L: float,
    c0: float,
    lP: float,
    gamma: float,
    use_planck_scale: bool,
    fit_alpha: bool,
    fit_phi: bool,
    max_iter: int,
    n_starts: int,
    seed: int | None,
    backend: Backend,
) -> tuple[dict, float]:
    rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs = _build_initial_coeffs(
        data.S_hat_cpu, n_coeff, r
    )
    alpha_val = float(np.log(1.0))

    bounds = _default_bounds(n_coeff, r, fit_alpha, fit_phi)
    x0 = _pack_params(alpha_val, rho_coeffs, P_coeffs, B_real_coeffs, B_imag_coeffs, phi_coeffs, fit_alpha, fit_phi)

    def objective(x: np.ndarray) -> float:
        alpha_val_i, rho_c, P_c, B_r, B_i, phi_c = _unpack_params(x, n_coeff, r, fit_alpha, fit_phi)
        lnL = loglike_et_qif_cuda(
            alpha_val_i,
            rho_c,
            P_c,
            B_r,
            B_i,
            phi_c,
            data,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            backend=backend,
        )
        return -float(lnL)

    rng = np.random.default_rng(seed)
    best_res = None
    best_fun = np.inf
    for start in range(max(1, n_starts)):
        if start == 0:
            x_start = x0
        else:
            jitter = rng.normal(scale=0.1, size=x0.shape)
            x_start = x0 + jitter
        res = minimize(objective, x_start, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter})
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_res = res

    if best_res is None:
        raise RuntimeError("Optimization failed to produce any result.")

    alpha_val_i, rho_c, P_c, B_r, B_i, phi_c = _unpack_params(best_res.x, n_coeff, r, fit_alpha, fit_phi)
    lnL = -float(best_res.fun)
    params = {
        "alpha_val": alpha_val_i,
        "rho_coeffs": rho_c,
        "P_coeffs": P_c,
        "B_real_coeffs": B_r,
        "B_imag_coeffs": B_i,
        "phi_coeffs": phi_c,
    }
    return params, lnL


def bootstrap_lr_cuda(
    data: QIFData,
    n_coeff: int,
    r: int,
    L: float,
    c0: float,
    lP: float,
    gamma: float,
    use_planck_scale: bool,
    fit_phi: bool,
    n_boot: int,
    refit: bool,
    seed: int,
    max_iter: int,
    n_starts: int,
    bootstrap_mode: str,
    backend: Backend,
) -> tuple[float, float, np.ndarray]:
    rng_cpu = np.random.default_rng(seed)
    rng_xp = _make_rng(backend, seed)

    env_params, lnL_env = fit_model_cuda(
        data, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
        fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed, backend=backend
    )
    qif_params, lnL_qif = fit_model_cuda(
        data, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
        fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed, backend=backend
    )
    lr_obs = 2.0 * (lnL_qif - lnL_env)

    Sigma_env = _build_sigma_stack_cuda(env_params, data, L, c0, lP, gamma, use_planck_scale, backend)

    lrs = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        xp = backend.xp
        S_hat_b = xp.zeros_like(data.S_hat_x)
        for k in range(len(data.m_eff)):
            m_eff_k = float(data.m_eff[k])
            if bootstrap_mode == "probabilistic":
                m_star = draw_m_star(m_eff_k, rng_cpu)
            elif bootstrap_mode == "floor":
                m_star = int(np.floor(m_eff_k))
            elif bootstrap_mode == "ceil":
                m_star = int(np.ceil(m_eff_k))
            else:
                m_star = int(np.round(m_eff_k))
            m_star = max(1, m_star)
            S_hat_b[k] = sample_covariance_from_sigma_xp(Sigma_env[k], m_star, rng_xp, xp)

        if refit:
            data_b = replace(data, S_hat_x=S_hat_b, S_hat_cpu=backend.asnumpy(S_hat_b))
            env_b, lnL_env_b = fit_model_cuda(
                data_b, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
                fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed + i + 1,
                backend=backend
            )
            qif_b, lnL_qif_b = fit_model_cuda(
                data_b, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
                fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=seed + i + 1,
                backend=backend
            )
        else:
            data_b = replace(data, S_hat_x=S_hat_b)
            lnL_env_b = loglike_et_qif_cuda(
                env_params["alpha_val"],
                env_params["rho_coeffs"],
                env_params["P_coeffs"],
                env_params["B_real_coeffs"],
                env_params["B_imag_coeffs"],
                env_params["phi_coeffs"],
                data_b,
                L=L,
                c0=c0,
                lP=lP,
                gamma=gamma,
                use_planck_scale=use_planck_scale,
                backend=backend,
            )
            lnL_qif_b = loglike_et_qif_cuda(
                qif_params["alpha_val"],
                qif_params["rho_coeffs"],
                qif_params["P_coeffs"],
                qif_params["B_real_coeffs"],
                qif_params["B_imag_coeffs"],
                qif_params["phi_coeffs"],
                data_b,
                L=L,
                c0=c0,
                lP=lP,
                gamma=gamma,
                use_planck_scale=use_planck_scale,
                backend=backend,
            )
        lrs[i] = 2.0 * (lnL_qif_b - lnL_env_b)

    pval = float(np.mean(lrs >= lr_obs)) if n_boot > 0 else 1.0
    return float(lr_obs), pval, lrs


def _prepare_data(
    S_hat: np.ndarray,
    m_eff: np.ndarray,
    freqs: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
    knots: np.ndarray,
    P_floor: np.ndarray,
    B_clip: np.ndarray,
    T_over_f2_avg: np.ndarray | None,
    T_over_fgamma_avg: np.ndarray | None,
    backend: Backend,
) -> QIFData:
    return QIFData(
        freqs=np.asarray(freqs),
        knots=np.asarray(knots),
        S_hat_x=backend.asarray(S_hat, dtype=complex),
        S_hat_cpu=np.asarray(S_hat),
        m_eff=np.asarray(m_eff),
        f_min_x=backend.asarray(f_min),
        f_max_x=backend.asarray(f_max),
        P_floor_x=backend.asarray(P_floor) if P_floor is not None else None,
        B_clip_x=backend.asarray(B_clip) if B_clip is not None else None,
        T_over_f2_avg_x=backend.asarray(T_over_f2_avg) if T_over_f2_avg is not None else None,
        T_over_fgamma_avg_x=backend.asarray(T_over_fgamma_avg) if T_over_fgamma_avg is not None else None,
    )


def _run_loglike_on_group_cuda(
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
    backend: Backend,
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

    data_obj = _prepare_data(
        S_hat,
        m_eff,
        freqs,
        f_min,
        f_max,
        knots,
        P_floor,
        B_clip,
        T_over_f2_avg,
        T_over_fgamma_avg,
        backend,
    )

    results: dict = {
        "group": group,
        "freq_bins": len(freqs),
    }

    if not fit:
        lnL = loglike_et_qif_cuda(
            np.log(1.0),
            rho_coeffs,
            P_coeffs,
            B_real_coeffs,
            B_imag_coeffs,
            phi_coeffs,
            data_obj,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            backend=backend,
        )
        results["loglike"] = float(lnL)
        return results

    env_params, lnL_env = fit_model_cuda(
        data_obj, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
        fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"], backend=backend
    )
    qif_params, lnL_qif = fit_model_cuda(
        data_obj, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
        fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"], backend=backend
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
        lr_obs, pval, lrs = bootstrap_lr_cuda(
            data_obj,
            n_coeff,
            r,
            L=L,
            c0=c0,
            lP=lP,
            gamma=gamma,
            use_planck_scale=use_planck_scale,
            fit_phi=fit_phi,
            n_boot=bootstrap_n,
            refit=bootstrap_refit,
            seed=group["gps"] % 100000,
            max_iter=max_iter,
            n_starts=n_starts,
            bootstrap_mode=bootstrap_mode,
            backend=backend,
        )
        results["baseline"]["lr_bootstrap"] = float(lr_obs)
        results["baseline"]["p_value"] = float(pval)
        results["baseline"]["p_se"] = float(np.sqrt(pval * (1.0 - pval) / max(1, bootstrap_n)))

    if stress_rank2:
        env_params2, lnL_env2 = fit_model_cuda(
            data_obj, n_coeff, 2, L, c0, lP, gamma, use_planck_scale,
            fit_alpha=False, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 2,
            backend=backend
        )
        qif_params2, lnL_qif2 = fit_model_cuda(
            data_obj, n_coeff, 2, L, c0, lP, gamma, use_planck_scale,
            fit_alpha=True, fit_phi=fit_phi, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 2,
            backend=backend
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
        env_params0, lnL_env0 = fit_model_cuda(
            data_obj, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
            fit_alpha=False, fit_phi=False, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 3,
            backend=backend
        )
        qif_params0, lnL_qif0 = fit_model_cuda(
            data_obj, n_coeff, r, L, c0, lP, gamma, use_planck_scale,
            fit_alpha=True, fit_phi=False, max_iter=max_iter, n_starts=n_starts, seed=group["gps"] + 3,
            backend=backend
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


def run_on_sample_data_cuda(
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
    backend: Backend | None = None,
) -> None:
    if backend is None:
        backend = Backend(np, "numpy", False)
    groups = _find_gwf_groups(data_root)
    if not groups:
        print(f"No .gwf files found under {data_root}.")
        return
    for group in groups:
        results = _run_loglike_on_group_cuda(
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
            backend=backend,
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


def run_synthetic_cuda(backend: Backend) -> None:
    data = _make_synthetic_data()
    data_obj = _prepare_data(
        data["S_hat"],
        data["m_eff"],
        data["freqs"],
        data["f_min"],
        data["f_max"],
        data["knots"],
        data["P_floor"],
        data["B_clip"],
        data["T_over_f2_avg"],
        None,
        backend,
    )
    lnL = loglike_et_qif_cuda(
        data["alpha_val"],
        data["rho_coeffs"],
        data["P_coeffs"],
        data["B_real_coeffs"],
        data["B_imag_coeffs"],
        data["phi_coeffs"],
        data_obj,
        data["L"],
        data["c0"],
        data["lP"],
        backend=backend,
    )
    print(f"synthetic loglike_et_qif={lnL:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QIF likelihood on ET MDC GWF samples (CUDA-capable).")
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"],
                        help="Array backend: auto uses CUDA if available, else CPU.")
    args = parser.parse_args()

    backend = get_backend(args.device)
    print(f"Backend: {backend.name}")

    if args.synthetic:
        run_synthetic_cuda(backend)
        return

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_root))
    if not os.path.isdir(base):
        print(f"Data root not found: {base}. Running synthetic demo.")
        run_synthetic_cuda(backend)
        return

    line_mask = _load_line_mask(args.line_mask) if args.line_mask else None
    transfer_data = None
    if args.transfer_csv:
        transfer_data = _load_transfer_csv(args.transfer_csv, is_sq=args.transfer_sq)

    run_on_sample_data_cuda(
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
        backend=backend,
    )


if __name__ == "__main__":
    main()
