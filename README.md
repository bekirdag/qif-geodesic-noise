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

## Findings so far (summary)

All runs were performed on **ET-MDC1 loudest sample sets**:

- BBH_snr_306, BBH_snr_344, BBH_snr_379, BBH_snr_387, BBH_snr_587
- BNS_snr_379

Key observations from CPU runs:

- **Smoke test (64 s / 64 bins, no fit)**: identical log-likelihood across all groups (~7.857e4).
- **Short fit (64 s / 64 bins, fit only)**: LR ~ -1.023e3 for BBH_snr_306.
- **Short fit w/ phase (128 s / 128 bins)**: LR ~ -4.640e3 for BBH_snr_306.
- **Full-data run (128 s / 128 bins)**: identical LR (~-4.640e3) across all groups.
- **Full-duration run (2048 s / 512 bins)**: identical LR (~-3.568e5) across all groups.

Interpretation:
- Under current settings, the geodesic-diffusion term is **not favored** (negative LR).
- Identical LR values across groups suggest either (a) the data windows are effectively identical in
  the statistics used, or (b) sensitivity is still limited at current settings.

Planned next-pass improvements:
- Increase resolution (`--max-bins`) where feasible.
- Apply line masks and response/transfer corrections.
- Test rank-2 environmental models and bootstrap modes.

## Validation results (GPU, RTX 3090)

These are user-reported validation runs recorded under `test-runs/`:

- **Resolution sweep (2048 s, r=1)**: LR remains negative and grows in magnitude as bins increase (128/256/512).
- **Bootstrap (n=50)**: p=1.0000 at 128 s / 128 bins (consistent with null under current settings).
- **Bootstrap (n=20, all groups)**: p=1.0000 across all groups at 128 s / 128 bins.
- **Stress tests**: rank-2 and phi-fixed variants match the baseline LR.
- **Line mask + transfer (template)**: LR remains negative and close to baseline.
- **Synthetic injection recovery**: positive LR (2.074083e+02) for injected alpha=1.0.
- **Synthetic injection sweep**: alpha=0.1/0.3/1.0 all yield positive LR (~2.07e+02), confirming recoverability (not a threshold).

## Manuscript and method

- `docs/manuscript.md` contains the full paper with citations and reference URLs.
- `docs/method.md` provides a concise, implementation-oriented summary of the pipeline.

# qif-geodesic-noise
