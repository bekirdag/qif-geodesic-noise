# 2026-07-02 v0.2 corrected pipeline — full validation runs (CPU, macOS arm64)

Environment: Python 3.11 venv (numpy/scipy/gwpy 4.0.1/lalsuite), MacBook CPU.
Data: fresh download of the full ET-MDC1 loudest manifest (all 25 gwf files verified
distinct by md5; sizes ~128.65 MB each).

## Root-cause postmortem of v0.1 "identical LR" results

Reproduced on synthetic data at realistic strain scale (1e-47 strain^2/Hz): two
DIFFERENT noise realizations returned byte-identical lnL_env = 1.1520000e+05
(= the analytic corner value: 32 bins x m=40 x 90) and identical negative LR with
alpha_hat pinned at the e^-20 bound. Cause: fixed parameter boxes/clips excluded
the physical scale of strain data (log P bounds (-30,30) + clip ±50 vs true ~-108;
log alpha bounds (-20,20) vs true ~-100; no-fit path evaluated alpha=1).
The v0.1 empirical section is withdrawn; the old data was never actually consumed
by the optimizer.

## Fixes applied (qif_v2.py, mirrored in qif_v2_cuda.py)

1. Data-adaptive bounds centered on measured scales (`_data_scales`, `_default_bounds`).
2. Overflow-only clips (±700) inside the likelihood and sigma stack.
3. Nested warm start (H1 from H0 optimum) -> LR >= 0 by construction.
4. 1-D profile scan over log-alpha to seed the signal start (log-space gradient
   vanishes at alpha ~ 0; a raw data-scale start overshoots the basin).
5. Cross-pollinated nested pair (`fit_nested_pair`) for symmetric optimizer effort.
6. Scale-aware multistart jitter; alpha=0 in the no-fit smoke test.
7. Welch bin edges computed before subsampling.
8. Vectorized Wishart likelihood (batched 3x3 Cholesky): ~40x faster; verified equal
   to the loop implementation to 13 digits.
9. Default n_coeff 6 -> 20: with 6 coefficients the spline cannot represent the PSD
   and the f^-2 term absorbs the misfit (spurious LR up to ~712 with uniform
   A_h_hat ~1.2e-47 in EVERY group; collapses to O(1) at n_coeff=20).

## Smoke test (no fit, alpha=0), 64 s / 64 bins — now data-dependent

BBH_snr_306 6.147608e+05 | BBH_snr_344 6.148809e+05 | BBH_snr_379 6.149090e+05
BBH_snr_387 6.149487e+05 | BBH_snr_587 6.147970e+05 | BNS 1001329152 6.147439e+05
BNS 1001331200 6.151794e+05 | BNS 1001333248 6.147525e+05

## Baseline fits, 128 s / 128 bins, n_coeff=20, r=1, fit-phi, max-iter 500, n-starts 2

BBH_snr_306 gps=1000245760 lr=8.184134e+00 A_h_hat=1.214e-47
BBH_snr_344 gps=1002150400 lr=1.243832e+00 A_h_hat=4.558e-48
BBH_snr_379 gps=1001558528 lr=3.737605e+00 A_h_hat=1.249e-47
BBH_snr_387 gps=1001622016 lr=6.369332e+00 A_h_hat=1.275e-47
BBH_snr_587 gps=1001619968 lr=5.460451e+00 A_h_hat=1.220e-47
BNS_snr_379 gps=1001329152 lr=5.152157e+01 A_h_hat=3.295e-47  <- CBC-contamination candidate
BNS_snr_379 gps=1001331200 lr=1.926016e+00 A_h_hat=4.780e-48
BNS_snr_379 gps=1001333248 lr=3.102998e+00 A_h_hat=1.186e-47

All LR >= 0 as required for a nested test. The loudest excursion is the first BNS
segment — expected: these segments contain loud CBC injections that produce real
correlated cross-channel power. Shakedown only; not statements about nature.

## Injection recovery (BBH_snr_306-conditioned resamples, 64 s / 64 bins, n_coeff=20)

Fisher sigma_F(A_h) = 7.218e-48 (naive, rho=0.5)
A_inj = 0        -> LR ~ 0        A_hat ~ 0
A_inj = 2 sigma  -> LR = 3.52     A_hat = 1.39e-47
A_inj = 4 sigma  -> LR = 0.20     A_hat = 5.32e-48
A_inj = 8 sigma  -> LR = 31.5     A_hat = 3.72e-47
A_inj = 16 sigma -> LR = 73.4     A_hat = 1.01e-46
A_inj = 32 sigma -> LR = 194.9    A_hat = 2.89e-46
A_inj = 64 sigma -> LR = 554.6    A_hat = 2.90e-46
A_inj = 128 sigma-> LR = 1406.9   A_hat = 7.83e-46
Empirical reliable-detection threshold ~8 sigma_F (~6e-47): nuisance-model
absorption costs ~1 order of magnitude vs naive Fisher.

## Profile-likelihood upper limit (BBH_snr_306, 128 s / 128 bins)

Unimodal deviance; max at A_h ~ 1.8e-47 (improvement LR ~ 8.0, consistent with the
baseline fit); UL95(A_h) = 2.24e-47 Hz^-1 (f0 = 1 Hz).

## Null stream + coherence (BBH_snr_306, 128 s)

Null stream (E1+E2+E3) amplitude = 1.742x single channel (sqrt(3) expected for
independent channel noise). Pairwise coherences at the 1/N_seg floor.

## Fisher sensitivity forecast (measured MDC1 PSD, rho=0.5, 10 Hz–Nyquist)

UL95(A_h): 1.50e-48 (128 s) | 3.76e-49 (2048 s) | 5.78e-50 (1 day) | 3.03e-51 (1 yr)
Existing bounds mapped to gamma=2: resonators 3.6e-29 (6 mHz), 2.5e-33 (>5 mHz);
TAMA-class 2e-35. Planck RW benchmark (alpha=1, L=1e4 m): A_h = 2.45e-36 —
currently allowed (TAMA reaches alpha~8); 1 yr of ET probes alpha ~ 1e-15.

## Bootstrap (config-matched, BBH_snr_306, 128 s / 128 bins, n=20, full refits)

LR_obs = 8.184134e+00, p = 0.0476 +/- 0.0476 (0/20 null draws exceeded LR_obs).
Mild excess consistent with CBC contamination of the loudest segments; not detection-grade.

## Bootstrap (BBH_snr_306, 64 s / 64 bins, n=40, full refits)

LR_obs = 2.0290e-01, p = 0.0732. Null draws: 85% boundary mass at ~0, positive tail to 1.106.
Draws and figure: docs/figures/bootstrap_results.json, docs/figures/fig_bootstrap.pdf.
