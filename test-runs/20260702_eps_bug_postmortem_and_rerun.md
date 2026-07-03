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

---

# 2026-07-03: v0.3.1 second referee round — conditioning stall found; "no detection power" RETRACTED

Prompted by a second set of referee questions on v0.3. Full results:
`test-runs/rerun_results_v031.json` (produced by the analytic-score suite).

## Defect 6: optimizer conditioning stall (the one that mattered scientifically)

Raw parameters span ~26 orders of magnitude (log-PSDs ~ -108, B coefficients
~1e-24, phases ~1). Against L-BFGS-B's identity initial Hessian, descent from
any distant start stalls hundreds of lnL units short of the optimum EVEN WITH
exact analytic gradients. Fix: optimize in scaled variables x = s*y (s = 1 for
log/logit/phase blocks, s = B_scale for B blocks). Measured on the 128-bin env
fit: +642 lnL units recovered; 4/5 independent seeds land on the identical
optimum to 4 decimals; restart scatter collapses 50-80 -> ~4-7 units.

Scientific consequence: the v0.3 "complete absorption / no detection power"
finding DOES NOT SURVIVE. Both hypotheses' fits had stalled symmetrically and
the guarded pairing reported the difference of two equally-unconverged optima
as zero. Corrected measurements (BBH_snr_306, 64 s / 64 log bins, rho=0.5):

- LR injection ladder (16 draws/amplitude): turn-on ~30 sigma_F; med Lambda =
  10.0 / 57.0 / 159.7 / 386.0 at 32/64/128/256 sigma_F; recovered amplitudes
  unbiased (ratio 0.9-1.2) above threshold.
- Null LR tail is Davies-inflated (free rho(f) spline): null draws reach
  Lambda = 22-23; bootstrap n=100 gives Lambda_95 = 4.53, boundary mass 0.52
  (was 1.00 with stalled fits — another signature). Nominal 2.71 cuts invalid.
- Asimov (noiseless) ladder through the SAME evaluator: Lambda_Asimov =
  0.006 / 0.17 / 49.3 / 393.6 / 1345 at 4/16/64/256/1024 sigma_F; noiseless
  crossing ~31 sigma_F, consistent with the noisy ladder. Absorption is
  98-99.9% of available deviance — overwhelming but NOT total.
- Profile UL @128s: profile now resolvable; long absorption plateau
  (ridge-and-dip <= ~9 units from basin discovery) then a steep wall;
  UL95(A_h) = 7.8e-47 Hz^-1 (per-path, rho=0.5). This is 10x WEAKER than the
  v0.3 "conservative" 7.7e-48 — the v0.3 stalled profile fits could not
  perform legitimate absorption of a fixed template, so the old limit was
  actually anti-conservative by ~10x.
- All 8 baseline groups remain null in both channels (LR <= 0.08; sign p in
  [0.14, 0.99]).

## Defect 7 (methodological): jitter-evaluator mismatch in absolute comparisons

The likelihood's diagonal jitter guard (1e-9 * peak PSD per bin) contributes
an O(100)-unit offset at lnL ~ 1e8 vs jitterless closed-form evaluations
(measured: 286.0 exactly reproduced by adding the jitter to the stack).
An early Asimov run mixed evaluators and produced an impossible result (a fit
landing ~600 units below its own warm start), caught by that internal bound.
Rule: every absolute lnL comparison must flow through the production evaluator,
including the generation of Asimov data (truth must include the jitter).

## New: sign-channel statistic (fit-free, gauge/convention-invariant)

T_sign = sum_k m_k^{3/2} Re(t_k)/(S11 S22 S33)_k with t = S12 S23 S31.
- Null moments verified: E[r] = 1/m^2 (+1%), sd(r) ~ 0.75 m^{-3/2}.
- Calibration is UNIVERSAL (diagonal-null distribution depends only on {m_k},
  by scale invariance of r) and conservative under the whole diag+rank-1+phase
  class (the sign theorem). Plug-in calibration under the FITTED env model is
  anti-conservative (measured false rate 0.56 at nominal 0.05) because the
  fitted rank-1 factor soaks sampling coherence into positively-biased-t
  structure. With the universal calibration: false rate 0.06 at null (n=16).
- Power: 0.19 / 1.00 / 1.00 at 64/128/256 sigma_F; 50% at ~84 sigma_F.
  ~3x slower than the corrected LR, but immune to nuisance absorption and
  optimizer misbehavior. Line robustness: a rank-1 line CANNOT fake Re t < 0
  (measured false rate 0.02 vs 0.45 clean); rank-2 lines give the predicted
  ~17%.
- Bartlett-decomposition Wishart sampler added (exact, O(1) per bin at any m).

## Also fixed

- Paper derivation normalization: spurious 1/2 in the v0.3 template equation;
  A_h is the PER-PATH amplitude; per-channel signal PSD = 2 A_h f^-gamma.
  Code was consistent throughout; no number changes.
- Analytic Wishart scores (loglike_et_qif_grad): verified against central FD
  (residual scales exactly as the FD probe's own f*eps_mach/h floor);
  ~13x faster; CUDA variant still on scale-matched FD (flagged, port pending).
- fisher @64s: naive 1.68e-48, profiled 2.01e-48 (x1.20, pinv gauge handling).

## Deliverables

- Paper v0.3.1: docs/et_geodesic_noise_paper_v0.3.tex/.pdf (21 pp), 4 figures
  regenerated.
- qif_v2.py: loglike_et_qif_grad, scaled optimization, sign_channel_stat,
  sign_channel_pvalue, sample_wishart_bartlett.
- scripts/check_analytic_scores.py: gradient regression test.
