#!/usr/bin/env python3
import argparse
import csv
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


def _parse_alpha_grid(text: str) -> list[float]:
    if not text:
        return []
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic QIF injection recovery test.")
    parser.add_argument("--n-f", type=int, default=16, help="Number of frequency bins.")
    parser.add_argument("--n-coeff", type=int, default=6, help="Number of spline coefficients.")
    parser.add_argument("--r", type=int, default=1, help="Environmental rank.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Injected alpha amplitude (phenomenological).")
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="",
        help="Comma-separated alpha values to sweep (overrides --alpha).",
    )
    parser.add_argument(
        "--alpha-logspace",
        type=float,
        nargs=3,
        metavar=("LOG10_MIN", "LOG10_MAX", "N"),
        help="Log10-spaced alpha sweep (overrides --alpha if --alpha-grid is not set).",
    )
    parser.add_argument("--m-eff", type=int, default=20, help="Effective averages per bin (integer).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--fit-phi", action="store_true", help="Fit calibration phases.")
    parser.add_argument("--max-iter", type=int, default=200, help="Max optimizer iterations.")
    parser.add_argument("--n-starts", type=int, default=3, help="Optimizer multi-starts.")
    parser.add_argument("--gamma", type=float, default=2.0, help="Spectral index for f^-gamma.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional CSV path to append results (header written if missing).",
    )
    args = parser.parse_args()

    if args.m_eff <= 0:
        raise SystemExit("--m-eff must be > 0")
    if args.alpha < 0:
        raise SystemExit("--alpha must be >= 0")

    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    if alpha_grid and any(a < 0 for a in alpha_grid):
        raise SystemExit("--alpha-grid values must be >= 0")

    alpha_logspace = None
    if args.alpha_logspace is not None:
        log10_min, log10_max, n_points = args.alpha_logspace
        n_points_int = int(n_points)
        if n_points_int <= 0:
            raise SystemExit("--alpha-logspace N must be > 0")
        alpha_logspace = np.logspace(float(log10_min), float(log10_max), n_points_int).tolist()

    if alpha_grid:
        alphas = alpha_grid
    elif alpha_logspace is not None:
        alphas = alpha_logspace
    else:
        alphas = [args.alpha]

    data = _make_synthetic_data(n_f=args.n_f, n_coeff=args.n_coeff, r=args.r, seed=args.seed)

    results = []
    for alpha in alphas:
        params = {
            "alpha_val": float(np.log(max(alpha, 1e-30))) if alpha > 0 else float("-inf"),
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

        _, lnL_env = fit_model(
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

        _, lnL_qif = fit_model(
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
        results.append((alpha, lnL_env, lnL_qif, lr))

    print("Synthetic injection recovery")
    print(f"  n_f={args.n_f} n_coeff={args.n_coeff} r={args.r} m_eff={args.m_eff} gamma={args.gamma}")
    for alpha, lnL_env, lnL_qif, lr in results:
        print(f"  alpha_injected={alpha}")
        print(f"  lnL_env={lnL_env:.6e}")
        print(f"  lnL_qif={lnL_qif:.6e}")
        print(f"  LR={lr:.6e}")

    if args.out_csv:
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "alpha_injected",
                    "lnL_env",
                    "lnL_qif",
                    "LR",
                    "n_f",
                    "n_coeff",
                    "r",
                    "m_eff",
                    "gamma",
                    "fit_phi",
                    "seed",
                ],
            )
            if write_header:
                writer.writeheader()
            for alpha, lnL_env, lnL_qif, lr in results:
                writer.writerow(
                    {
                        "alpha_injected": alpha,
                        "lnL_env": lnL_env,
                        "lnL_qif": lnL_qif,
                        "LR": lr,
                        "n_f": args.n_f,
                        "n_coeff": args.n_coeff,
                        "r": args.r,
                        "m_eff": args.m_eff,
                        "gamma": args.gamma,
                        "fit_phi": bool(args.fit_phi),
                        "seed": args.seed,
                    }
                )


if __name__ == "__main__":
    main()
