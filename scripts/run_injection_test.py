#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qif_v2 import (
    _make_synthetic_data,
    _build_sigma_stack,
    sample_covariance_from_sigma,
    fit_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic QIF injection recovery test.")
    parser.add_argument("--n-f", type=int, default=16, help="Number of frequency bins.")
    parser.add_argument("--n-coeff", type=int, default=6, help="Number of spline coefficients.")
    parser.add_argument("--r", type=int, default=1, help="Environmental rank.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Injected alpha amplitude (phenomenological).")
    parser.add_argument("--m-eff", type=int, default=20, help="Effective averages per bin (integer).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fit-phi", action="store_true", help="Fit calibration phases.")
    parser.add_argument("--max-iter", type=int, default=200, help="Max optimizer iterations.")
    parser.add_argument("--n-starts", type=int, default=3, help="Optimizer multi-starts.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Spectral index for f^-gamma.")
    args = parser.parse_args()

    if args.m_eff <= 0:
        raise SystemExit("--m-eff must be > 0")
    if args.alpha < 0:
        raise SystemExit("--alpha must be >= 0")

    data = _make_synthetic_data(n_f=args.n_f, n_coeff=args.n_coeff, r=args.r, seed=args.seed)

    params = {
        "alpha_val": float(np.log(max(args.alpha, 1e-30))) if args.alpha > 0 else float("-inf"),
        "rho_coeffs": data["rho_coeffs"],
        "P_coeffs": data["P_coeffs"],
        "B_real_coeffs": data["B_real_coeffs"],
        "B_imag_coeffs": data["B_imag_coeffs"],
        "phi_coeffs": data["phi_coeffs"],
    }

    Sigma = _build_sigma_stack(
        params,
        data["freqs"],
        data["f_min"],
        data["f_max"],
        data["knots"],
        data["P_floor"],
        data["B_clip"],
        data["L"],
        data["c0"],
        data["lP"],
        gamma=args.gamma,
        use_planck_scale=False,
        T_over_f2_avg=data["T_over_f2_avg"],
        T_over_fgamma_avg=None,
    )

    rng = np.random.default_rng(args.seed + 1)
    S_hat = np.zeros_like(Sigma)
    for k in range(Sigma.shape[0]):
        S_hat[k] = sample_covariance_from_sigma(Sigma[k], args.m_eff, rng)

    m_eff = np.full((args.n_f,), float(args.m_eff))

    env_params, lnL_env = fit_model(
        S_hat,
        m_eff,
        data["freqs"],
        data["f_min"],
        data["f_max"],
        data["knots"],
        data["P_floor"],
        data["B_clip"],
        args.n_coeff,
        args.r,
        L=data["L"],
        c0=data["c0"],
        lP=data["lP"],
        gamma=args.gamma,
        use_planck_scale=False,
        T_over_f2_avg=data["T_over_f2_avg"],
        T_over_fgamma_avg=None,
        fit_alpha=False,
        fit_phi=args.fit_phi,
        max_iter=args.max_iter,
        n_starts=args.n_starts,
        seed=args.seed,
    )

    qif_params, lnL_qif = fit_model(
        S_hat,
        m_eff,
        data["freqs"],
        data["f_min"],
        data["f_max"],
        data["knots"],
        data["P_floor"],
        data["B_clip"],
        args.n_coeff,
        args.r,
        L=data["L"],
        c0=data["c0"],
        lP=data["lP"],
        gamma=args.gamma,
        use_planck_scale=False,
        T_over_f2_avg=data["T_over_f2_avg"],
        T_over_fgamma_avg=None,
        fit_alpha=True,
        fit_phi=args.fit_phi,
        max_iter=args.max_iter,
        n_starts=args.n_starts,
        seed=args.seed,
    )

    lr = 2.0 * (lnL_qif - lnL_env)

    print("Synthetic injection recovery")
    print(f"  n_f={args.n_f} n_coeff={args.n_coeff} r={args.r} m_eff={args.m_eff} gamma={args.gamma}")
    print(f"  alpha_injected={args.alpha}")
    print(f"  lnL_env={lnL_env:.6e}")
    print(f"  lnL_qif={lnL_qif:.6e}")
    print(f"  LR={lr:.6e}")


if __name__ == "__main__":
    main()
