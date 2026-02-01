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
# Example (choose the CuPy build that matches your CUDA version)
pip install cupy-cuda11x
```

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

## Manuscript and method

- `docs/manuscript.md` contains the full paper with citations and reference URLs.
- `docs/method.md` provides a concise, implementation-oriented summary of the pipeline.

# qif-geodesic-noise
