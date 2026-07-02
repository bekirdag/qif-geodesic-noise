# 2026-07-02 (later the same day): third optimizer bug found and fixed; v0.2 numbers superseded

Discovered while answering the referee questions in `docs/quesitons.md`
(see `docs/answers_to_questions_v0.2.md` for full context and all answers).

## The bug (finite-difference eps scale mismatch)

`scipy.optimize.minimize(method="L-BFGS-B")` defaults to ABSOLUTE finite-difference
steps eps ~ 1.5e-8. The environmental factor coefficients B have physical scale
~1e-24 (sqrt of strain PSD). A 1.5e-8 probe step inflates B B^H by ~32 orders of
magnitude; the numerical gradient in those directions is ~1e24 (garbage); every line
search fails; L-BFGS-B exits at nit=0, status=2 ABNORMAL with the start point
returned untouched.

Measured on a v0.2-configuration fit (BBH_snr_306, 64 s / 64 bins, N_c=20):
- nit=0, status=2 ABNORMAL, nfev=5061, 0 of 240 parameters moved
- max|grad| = 1.24e24
- fitted "PSD" = flat initialization (coefficient spread 0.00; data dynamic range x76)
- per-bin Wishart traces 0.2-7.1 (perfect fit = 3)
- lnL 614760.8 vs 618725.6 for a perfect diagonal fit: ~3965 lnL units unclaimed

Detection signature that exposed it: LR byte-identical to 16 digits across spline
bases of DIFFERENT dimension (N_c = 20/28/36) and across r = 1 vs r = 2. Identical
statistics across different model dimensions = the fits never moved. This is the
inverse of the usual stability check and joins the v0.1 postmortem rules.

Consequence: all v0.2 LR structure came from the gradient-free 1-D profile scan over
log A_h; the 240-dimensional joint fits were inert. Every v0.2 headline number
(Table 1, bootstrap p, injection threshold, profile UL) is superseded.

## The fix (qif_v2.py, mirrored in qif_v2_cuda.py)

1. Per-parameter eps vector: 1e-4 (log/logit/phase blocks), 1e-4 * B_scale (B blocks).
2. maxfun raised to 2*max_iter*(dim+1) (scipy default 15000 truncates a 240-dim fit
   after ~60 gradient evaluations).
3. P-spline initialization by least squares on the log periodogram
   (_spline_design_matrix + lstsq in _build_initial_coeffs) instead of flat median.

Verification (same fit): lnL = 618685.8 (within 40 of the diagonal bound), P splines
span 4.7 e-folds tracking the data, per-bin traces 2.4-3.1, ~10 s per env fit.

## Two further defects found during re-validation

4. Nested-pair symmetrization gap: one cross-pollination round is insufficient — the
   alternative refit can enter a better shared-nuisance basin after the null's last
   refit (observed: spurious LR=19.8 with A_hat at the floor). Fixed: iterated
   cross-pollination + a direct (fit-free) evaluation of the null likelihood at the
   alternative's nuisance solution as an exact guard. New rule: positive LR with
   A_hat at the boundary is a pipeline error.
5. Data-discarding bin layout: max_bins SUBSAMPLED every ~250th native Welch bin
   (linear grid): 99.6% of spectral data discarded, one analysis bin below 100 Hz.
   Replaced with log-spaced block averaging of all native bins with Hann-correlation-
   corrected m_eff (_compute_welch_csd_matrix bin_spacing="log" default) and
   log-spaced spline knots (_open_log_knots).

## Definitive re-run (v0.3 pipeline, log grid) — rerun_results_log.json

- Baseline, 8 groups @128 s/128 log bins: ALL LR = 0.000 (A_hat at floor). The v0.2
  excesses (LR up to 8.2; BNS "outlier" LR=51.5) were nuisance misfit in the single
  under-resolved low-f bin; the outlier collapses to 0.000.
- Injection ladder @64 s (sigma_F = 1.781e-48 naive): NO recovery at any amplitude up
  to 256 sigma_F = 4.6e-46 Hz^-1; LR = 0.000 throughout. Verified honest: fitted
  signal-free model exceeds lnL of the generating truth (+29 at 256 sigma_F).
  Mechanism: rank-1 tangent space covers 5/6 off-diagonal dof; manifold relocation
  shadows the rest. The unconstrained LR search is an upper-limit machine, not a
  detection machine.
- fmin sweep: LR=0 at all cutoffs; sigma_F degrades only x1.9 from 5->30 Hz (log bins).
- rank-2, n_coeff {20,28,36}: all LR=0; lnL differs across bases (fits alive).
- Bootstrap n=20 @64 s: all draws ~0; boundary mass ~100%; p=0.95 (uninformative —
  statistic degenerate at the boundary; null-side face of the no-power finding).
- Profiled Fisher @64 s: naive 1.781e-48, locally profiled 2.108e-48 (x1.2) — local
  quantity; global absorption is what injections measure.
- Profile UL @128 s (continuation warm starts): optimizer-noise-limited. Measured
  nuisance-basin scatter ~76 lnL units (lnL ~ 1.7e8; FD gradient noise ~0.4/component).
  Conservative UL95(A_h) ~ 7.7e-48 Hz^-1 (threshold 2.71 + 2x scatter, referenced to
  best null); statistical crossing ~2e-48 not certifiable without analytic gradients.
  Verified the profile plateau was nuisance-optimization accumulation: null at the
  best profile nuisances beats the profile value by 12.9 (data disfavor the template).

## Deliverables

- Paper v0.3: docs/et_geodesic_noise_paper_v0.3.pdf (15 pp), .tex, 4 new figures
  (fig_injection_v03, fig_bootstrap_v03, fig_profile_ul_v03, fig_forecast_v03).
- Answers to the referee questions: docs/answers_to_questions_v0.2.md (all 27).

## Scientific consequences

1. Baseline groups: all LR collapse to O(0-5); the v0.2 "mild excesses" (LR 8.18,
   and the BNS outlier LR 51.5) were nuisance-misfit artifacts the crippled optimizer
   could not remove, NOT CBC cross-power.
2. Injection recovery: with a WORKING nuisance fit, the flexible instrument+environment
   model absorbs the geodesic template's diagonal entirely. The v0.2 "8 sigma_F
   effective threshold" was an artifact of frozen nuisance fits; the honest threshold
   (sign-obstruction channel only) is far higher - see the new ladder.
3. Identifiability now rests on the gauge-invariant triple-product sign obstruction
   Re(S12 S23 S31) < 0, which no diagonal + rank-1 + phase model can produce
   (theorem, numerically verified), and which dies at rank 2 (Ledermann bound:
   diag + rank-2 reproduces ANY 3x3 Hermitian covariance).
