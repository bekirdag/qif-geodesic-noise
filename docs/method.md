# Method Overview (QIF / Geodesic-Diffusion Noise Search)

This document is a practical, implementation-oriented description of the method used in `qif_v2.py`.
It is intended to help a researcher understand the analysis flow and reproduce the results without
reading the full manuscript end-to-end. For theory and citations, see `docs/manuscript.md`.

## 1) Goal

We test a **phenomenological** stochastic noise model for the Einstein Telescope (ET): a strain
power spectral density (PSD) that scales as `f^-gamma` (random-walk is `gamma=2`). The goal is to
estimate whether a **geodesic-diffusion-like** component is supported by the data beyond
instrument and environmental correlations.

## 2) Data and inputs

- ET-MDC1 loudest sample sets (3 channels: E1/E2/E3; sometimes E0 included in the archive)
- Each `.gwf` file is a 2048 s segment with channel data
- Optional: line masks (to exclude known lines), transfer functions (to map calibration)

## 3) Preprocessing and spectral estimation

1. Read E1/E2/E3 time series from `.gwf` files.
2. Apply Welch segmentation: window, overlap, and FFT to compute cross-spectral matrices.
3. Average segments and frequencies into bins, producing a 3x3 cross-spectral estimate per bin.
4. Symmetrize the estimate to enforce Hermitian structure.

## 4) Model components

For each frequency bin k, the model covariance is:

Sigma_k = G_k (Sigma_inst_k + Sigma_env_k + Sigma_sig_k) G_k^H

- **Instrument noise (Sigma_inst)**:
  - Diagonal PSDs P_i(f) for each channel.
  - Each P_i is parameterized by cubic B-splines in log-space.

- **Environmental coherence (Sigma_env)**:
  - Low-rank factor model: B_k B_k^H.
  - B_k spline-smoothed across frequency.
  - Rank is configurable (r=1 default; r=2 optional).

- **Signal (Sigma_sig)**:
  - Geodesic-diffusion PSD with amplitude A_h and index gamma.
  - Mapped into the 3x3 channel covariance via a response template.
  - The default implementation uses a symmetric template; response matrix can be upgraded.

- **Calibration/phase (G_k)**:
  - Optional phase corrections for channels 2 and 3 (channel 1 is gauge-fixed).

## 5) Likelihood model

The binned cross-spectral matrix S_hat,k is modeled as a **complex Wishart** random matrix with
an **effective** number of averages m_eff,k (overlap-corrected). The log-likelihood is the sum
across frequency bins of the complex-Wishart log-density.

This is an approximation and must be validated with time-domain simulations. The code includes
an overlap-aware estimate for m_eff,k.

## 6) Hypothesis testing

We compare two nested models:

- H_env: instrument + environmental coherence
- H_sig: instrument + environmental coherence + geodesic-diffusion signal

After fitting both models, we compute the likelihood ratio:

LR = log L(H_sig) - log L(H_env)

A negative LR indicates the signal term does not improve the fit under current settings.

## 7) Optimization and fitting

- Parameters are optimized with bounded numerical minimization.
- Multi-start optimization is used to reduce local-minimum risk.
- Optional phase parameters can be enabled (`--fit-phi`).
- Optional Planck-scale mapping can be toggled (`--use-planck-scaling`), but the default is a
  phenomenological amplitude A_h.

## 8) Bootstrap significance (optional)

A parametric bootstrap is implemented to calibrate LR under the null:

1. Fit H_env to the data.
2. Generate synthetic cross-spectral matrices under H_env.
3. Refit H_env and H_sig on each bootstrap sample.
4. Compute the empirical tail probability for LR.

Because the bootstrap is done per frequency bin (not full time-domain), results should be
validated against time-domain simulations.

## 9) Outputs

The script reports:

- Best-fit parameters (when fitting is enabled)
- Log-likelihoods and LR per dataset group
- Optional bootstrap p-values

## 10) Practical tips

- Start with short `--max-seconds` and small `--max-bins` for tuning.
- Use line masks and transfer functions once the pipeline is stable.
- Increase `--n-starts` and `--max-iter` for more stable fits.
- If LR is consistently negative, try more starts and consider warm-starting the signal fit from the env-only solution (alpha ~ 0).
- For GPU use, run `qif_v2_cuda.py` and set `--device auto` or `--device gpu`.


## 11) Validation checklist

Use the scripts in `scripts/` to confirm:

- Resolution sensitivity (sweep `--max-bins`).
- When comparing different `--max-bins`, also report LR per bin (LR / N_bins) to normalize for bin count.
- Line masks and transfer functions are applied correctly.
- Bootstrap p-values behave as expected under the null.
- Rank-2 and calibration-variant stress tests are stable.
- Synthetic injection recovery yields positive LR when the signal is present; for threshold curves sweep alpha over orders of magnitude.

All logs should be saved under `test-runs/`.
