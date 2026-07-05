# QIF / Geodesic-Diffusion Noise Search (ET-MDC1)

This repository contains our **phenomenological** geodesic-diffusion noise search for the Einstein Telescope (ET). It includes:

- The manuscript (`docs/manuscript.md`)
- A detailed method description (`docs/method.md`)
- The CPU pipeline (`qif_v2.py`) and CUDA-capable variant (`qif_v2_cuda.py`)
- A data download script for the **exact ET-MDC1 loudest samples** used in our runs

A researcher should be able to clone this repo, download the data with one command, and run the
analysis immediately.

## Repository layout

- `qif_v2.py` - primary CPU analysis implementation
- `qif_v2_cuda.py` - CUDA-capable variant (auto-detects GPU if available)
- `qif_likelihood.py` - convenience entry that re-exports `qif_v2`
- `docs/manuscript.md` - full manuscript with citations and reference URLs
- `docs/method.md` - implementation-focused method overview
- `scripts/download_data.sh` - download ET-MDC1 "loudest" sample data
- `data_manifest.tsv` - list of dataset files used in our runs
- `data/` - output directory for downloaded `.gwf` files

## Requirements

Python 3.10+ is recommended.

Minimal dependencies:

```
pip install -r requirements.txt
```

GPU (optional, NVIDIA CUDA only):

```
# Choose the CuPy build that matches your CUDA runtime
# - CUDA 11.x: cupy-cuda11x
# - CUDA 12.x: cupy-cuda12x
pip install cupy-cuda12x
```

CUDA requirements:
- You must have a compatible NVIDIA driver and CUDA runtime libraries installed.
- Check your CUDA version with `nvidia-smi` (look for "CUDA Version").
- If you see `libnvrtc.so` errors, install the matching CUDA runtime (or switch to the correct CuPy wheel).

## Downloading the ET-MDC1 loudest samples

We ran the analysis on the **ET-MDC1 loudest sample sets** (BBH and BNS). The data are hosted on the
ET OSDF Origin HTTP server.

Download **all** files used in our runs with:

```
./scripts/download_data.sh
```

This pulls the datasets listed in `data_manifest.tsv` into `data/<dataset>/`.

Notes:
- Base URL: `http://et-origin.cism.ucl.ac.be/`
- You can override it with `BASE_URL` if needed.

Example override:

```
BASE_URL=http://et-origin.cism.ucl.ac.be ./scripts/download_data.sh
```

## Running the analysis (CPU)

Smoke test (fast):

```
python qif_v2.py --data-root data --max-seconds 64 --max-bins 64
```

Tuned fit (short run):

```
python qif_v2.py --data-root data --max-seconds 128 --max-bins 128 --fit --fit-phi --max-iter 120 --n-starts 2
```

Full-duration run (2048 s, downsampled):

```
python qif_v2.py --data-root data --max-seconds 2048 --max-bins 512 --fit --fit-phi --max-iter 120 --n-starts 2
```

## Running the analysis (GPU)

The CUDA variant mirrors the CPU pipeline, but will use CuPy if available.

Auto-detect:

```
python qif_v2_cuda.py --data-root data --device auto --max-seconds 128 --max-bins 128 --fit --fit-phi
```

Force GPU:

```
python qif_v2_cuda.py --data-root data --device gpu --max-seconds 128 --max-bins 128 --fit --fit-phi
```

## Adjustment and fine-tuning logic

Use short runs to tune hyperparameters, then scale up:

- Start small: lower `--max-seconds` and `--max-bins` to validate IO and fit stability.
- Increase resolution: raise `--max-bins` to test sensitivity and stability.
- Optimize fits: increase `--max-iter` and `--n-starts` if the fit is unstable.
- Control leakage: adjust `--nperseg-seconds` and `--overlap` (Welch settings).
- Robustness: add `--bootstrap`, `--stress-rank2`, and `--calib-variants`.
- Data hygiene: apply `--line-mask` and `--transfer-csv` when available.

Example tuning sequence:

```
# Fast smoke test
python qif_v2_cuda.py --data-root data/BBH_snr_306 --device auto --max-seconds 64 --max-bins 64

# Short fit with phases
python qif_v2_cuda.py --data-root data/BBH_snr_306 --device auto --max-seconds 128 --max-bins 128 --fit --fit-phi --max-iter 120 --n-starts 2

# Full-duration, higher resolution
python qif_v2_cuda.py --data-root data --device auto --max-seconds 2048 --max-bins 512 --fit --fit-phi --max-iter 120 --n-starts 2
```

## Validation scripts

The `scripts/` directory includes helper scripts that implement the recommended validation checks:

- `scripts/run_sensitivity_sweep.sh` - sweep `--max-bins` to test resolution sensitivity
- `scripts/run_bootstrap.sh` - compute bootstrap p-values
- `scripts/run_stress_tests.sh` - rank-2 and calibration-variant stress tests
- `scripts/run_line_mask_transfer.sh` - apply line masks and transfer functions (templates provided)
- `scripts/run_injection_test.py` - synthetic injection recovery test
- `scripts/run_validation_suite.sh` - runs the full set in sequence

All scripts write logs to `test-runs/` in timestamped folders.

## Visualization / figures

Plotting utilities live in `scripts/plot_paper_figures.py`. Install plotting deps once:

```
pip install -r requirements-plot.txt
```

Generate validation plots from existing logs:

```
python scripts/plot_paper_figures.py validation --runs-dir test-runs --out-dir figures
```

This will attempt to write:
- `figures/lr_vs_bins.png`
- `figures/lr_per_bin.png`
- `figures/lr_by_dataset.png`
- `figures/bootstrap_p_values.png`
- `figures/stress_variants.png`

Generate a model PSD plot:

```
python scripts/plot_paper_figures.py psd --out figures/model_psd.png
```

Generate an injection sweep CSV and plot it:

```
python scripts/run_injection_test.py --alpha-logspace -8 2 15 --out-csv test-runs/injection_sweep.csv
python scripts/plot_paper_figures.py injection --csv test-runs/injection_sweep.csv --out figures/injection_curve.png --logx
```

Notes:
- `--logx` auto-skips non-positive alpha values.
- Use `--symlog` instead of `--logx` if you need to include alpha=0.

## Findings (v0.2, corrected pipeline)

> **CURRENT (2026-07-05): v0.3.3.** Fourth referee round (edge cases): the sign
> channel's **T_obs^{-1/2} scaling is demonstrated** (50% multipliers 69/64 σ_F at
> 64/128 s; the coarse "84" is refined to ~70) and the **two-channel discovery
> reach (~70 σ_F(T))** is on the forecast figure; production-tail validation shows
> **zero events at the 10⁻³ quantile** in 3×10⁴ coherent-null and 2×10⁴
> line-forest draws (the single-bin tail concern does not propagate to the
> aggregate); the line-forest **stress envelope** (10× density/power, combs) holds
> at false rate 0.000; cross-block Hann leakage is bounded (worst 0.11 on the
> 128-bin grid, ∝1/n²); and a real seeding weakness found by measurement is fixed:
> `fit_model` now scans **(amplitude, ρ) jointly**, restoring detection efficiency
> for frequency-decaying κ(f) signals (2/6 → 4/6 at 64 σ_F, unbiased recovery).
> Numbers: `test-runs/rerun_results_v033.json`; answers:
> `docs/answers_to_questions_v0.3.2.md`.

> **v0.3.2 (2026-07-03).** A third referee round produced boundary
> measurements, no retractions: the sign channel's universal calibration gains a
> measured finite-m validity boundary and a fit-free **coherence gate** (worst-case
> tail false rate 0.24 on strongly coherent rank-1 nulls; MDC1 passes the gate
> everywhere); an isotropic **SGWB also gives Re t < 0** (triangle correlation −1/2),
> so the sign channel establishes non-instrumental correlated power, not geodesic
> origin; a joint CBC-foreground amplitude costs **×2.45** in profiled Fisher (no
> collapse); LR optimization noise is negligible (Λ = 29.7 ± 0.2 at 32 σ_F); UL
> coverage first check 17/20; all sensitivity curves are labeled **path-template
> forecasts** (strain-reading forecast provisional pending full-response
> injections). Numbers: `test-runs/rerun_results_v032.json`; answers:
> `docs/answers_to_questions_v0.3.1.md`.

> **SUPERSEDED AGAIN (2026-07-03) — see v0.3.1:** a second referee round exposed an
> optimizer **conditioning stall** (raw parameters span ~26 orders of magnitude;
> fits stalled hundreds of lnL units short of their optima even with exact
> gradients). With closed-form analytic Wishart scores **and** scale-normalized
> optimization variables, the v0.3 "no detection power / complete absorption"
> finding is **retracted**: the LR test detects injections above ~30 sigma_F with
> unbiased amplitude recovery, the noiseless Asimov ladder confirms the threshold,
> and the profile upper limit becomes statistics-limited at
> **UL95(A_h) = 7.8e-47 Hz^-1** (128 s, rho=0.5, per-path convention) — 10x weaker
> than the v0.3 "conservative" limit, which stalled fits had made anti-conservative.
> v0.3.1 also adds a fit-free, convention-invariant **sign-channel statistic** with
> a universal conservative calibration (50% detection at ~84 sigma_F, immune to
> nuisance absorption), fixes a normalization inconsistency in the derivation
> (A_h is the per-path amplitude), and documents everything in
> **`docs/et_geodesic_noise_paper_v0.3.pdf`** (v0.3.1, 21 pp), Appendix "Revision
> record III". Definitive numbers: `test-runs/rerun_results_v031.json`; narrative:
> `test-runs/20260702_eps_bug_postmortem_and_rerun.md`.

> **SUPERSEDED (2026-07-02, later the same day) — see v0.3:** answering the referee
> questions in `docs/quesitons.md` exposed three further pipeline defects (absolute
> finite-difference steps vs. 1e-24-scale parameters; insufficient nested-fit
> symmetrization; a bin layout that discarded 99.6% of the spectral data). All v0.2
> numbers below are superseded. Full record:
> `test-runs/20260702_eps_bug_postmortem_and_rerun.md` and
> `docs/answers_to_questions_v0.2.md`.

**The v0.1 results (identical negative LR across all groups, bootstrap p=1.0) are
withdrawn.** They were artifacts of scale-inappropriate parameter bounds/clips that
pinned every fit at a data-independent corner solution; the optimizer never consumed
the data. See `test-runs/20260702_v02_corrected_pipeline.md` for the full postmortem
and `docs/et_geodesic_noise_paper_v0.2.pdf` (Appendix A) for the write-up.

v0.2 fixes (in `qif_v2.py`, mirrored in `qif_v2_cuda.py`): data-adaptive parameter
bounds, overflow-only clips, warm-started nested fits with cross-pollination
(`fit_nested_pair`, guarantees LR >= 0), 1-D profile-scan seeding of the signal
amplitude, scale-aware multistarts, vectorized Wishart likelihood (~40x faster),
`n_coeff` default raised to 20 (too few spline coefficients let the f^-2 term absorb
PSD misfit and fake a signal).

Key v0.2 results on ET-MDC1 loudest sets (CPU, 128 s / 128 bins, n_coeff=20):

- Per-group LR now non-negative, data-dependent, O(1-8) for 7/8 groups; the first
  BNS_snr_379 segment shows LR=51.5 (expected CBC contamination of "loudest" data).
- Bootstrap (BBH_snr_306): p=0.073 at 64 s; p=0.048±0.048 at 128 s (config-matched).
- Injection recovery on real-data-conditioned resamples: reliable from ~8x the naive
  Fisher scale (~6e-47 Hz^-1 at 64 s); recovered amplitude tracks injection.
- Profile-likelihood upper limit: A_h(95%) = 2.2e-47 Hz^-1 (BBH_snr_306, 128 s).
- Null stream (E1+E2+E3) = sqrt(3) x single channel: MDC1 channel noise independent.
- Fisher forecast: A_h(95%) ~ 3.8e-49 Hz^-1 per 2048 s, ~3e-51 Hz^-1 per year —
  12-16 orders below existing bounds; the Planck random-walk benchmark (alpha=1)
  is currently allowed and would be decisively tested.

These runs use deliberately CBC-loud synthetic data and are a pipeline shakedown,
not statements about nature.

## Manuscript and method

- `docs/manuscript.md` contains the full paper with citations and reference URLs.
- `docs/method.md` provides a concise, implementation-oriented summary of the pipeline.

# qif-geodesic-noise
