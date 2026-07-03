# Answers to the second-round (v0.3) referee questions → v0.3.1

Every item below is resolved in `docs/et_geodesic_noise_paper_v0.3.pdf` (v0.3.1,
2026-07-03). Definitive numbers: `test-runs/rerun_results_v031.json`. The five
"sticky questions" (S1–S5) and the fourteen numbered questions (Q1–Q14) are
answered together where they overlap.

**Headline change forced by this round:** fixing the optimizer conditioning
(Q14/S4 follow-through) **overturned v0.3's central negative finding**. The
unconstrained-nuisance LR search does have detection power — threshold
~30 σ_F at 64 s with unbiased amplitude recovery — and the v0.3 "complete
absorption" result is retracted as the third optimizer artifact (paper
Appendix "Revision record III"). The profile UL correspondingly *weakens* 10×
to 7.8e-47 Hz⁻¹: the old "conservative" limit was anti-conservative because
stalled fits could not perform legitimate absorption of a fixed template.

## S1 / Q3 — Why no fit-free sign statistic? Define it.
**Built.** Paper §"The sign channel as a statistic": T_sign = Σ m^{3/2}
Re(t)/(S11S22S33), t = S12S23S31. Null moments verified (E[r]=1/m², to 1%).
Calibration is **universal** (diagonal-null distribution depends only on
{m_k}, by scale invariance) and **provably conservative** under the whole
diag+rank-1+phase class via the sign theorem. Measured en route: the naive
plug-in bootstrap under the *fitted* env model is anti-conservative (false
rate 0.56 at nominal 0.05) — the fitted rank-1 factor soaks sampling
coherence into positively-biased-t structure. Power: 50% at ~84 σ_F (64 s);
0.19/1.00/1.00 at 64/128/256 σ_F; false rate 0.06 at null. It is now a
required coincidence condition in the pre-declared detection rule. It is
~3× slower than the corrected LR, so it is the *confirmation* channel, not
the sole discovery engine — the LR's power came back (see headline).

## S2 / (reading crisis) — Is ET DOA under the path-length reading?
**Yes, for the α=1 benchmark — stated bluntly in §Bounds.** Tabletop
resonators win by (L_ET/L_exp)² ~ 1e10 under that reading. The ET case rests
on the strain reading; the reading-robust science case is reframed as a
two-baseline measurement of the L-scaling exponent η in S_δℓ ∝ L^{2η}
(η=0 path-length, η=1 strain), with ET as the long-baseline anchor.

## S3 / Q4 / Q5 — Rank-2 reality; what justifies rank 1; is one triangle enough?
Paper §Stochastic-background confusion, expanded. Environmental rank is an
*empirical* question: the plan is witness-channel coherence matrices
measuring the coupled environmental subspace rank band by band; any band with
measured rank ≥2 is excluded from sign-channel discovery (ULs unaffected —
their validity proof never references environmental rank). A single
uninstrumented triangle supports upper limits and the conservative sign
channel but not an unconditional discovery claim; the minimal extra
information ladder (fixed templates → witnesses → second detector →
short-baseline anchor) is spelled out.

## S4 / Q14 / Q9 — Analytic gradients mandatory; is the UL optimizer-stable?
**Done, and it changed the science.** Closed-form Wishart scores
(`loglike_et_qif_grad`), verified against central FD (residual scales exactly
as the FD probe's own cancellation floor), ~13× faster. Second, independent
fix found in the process: **scale-normalized optimization variables** (raw
parameters span 26 orders of magnitude; identity-Hessian L-BFGS-B stalled
+642 lnL units short at 128 bins). Restart scatter: 50–80 → ≤7 lnL units;
4/5 independent seeds land on the *identical* optimum. The UL is now
statistics-limited; headline table, injection ladder, bootstrap (n=100),
Fisher, and profile UL all reproduced on the corrected optimizer.

## S5 — Does a joint CBC-foreground + geodesic fit leave identifiable space?
Parameter counting in §SGWB: *unconstrained* rank 2 absorbs everything (never
profile over it); the production foreground term is S_gw(f)Γ with Γ fixed by
geometry and S_gw pinned to the astrophysical shape family (2–3 parameters).
Against that, the geodesic term keeps three handles: spectral index
(f^{-7/3} vs f^{-2} differ by f^{1/3} ≈ ×4.6 over the band), null-stream
population ∝(1−ρ), and rank 3 vs rank 2. Degenerate only at ρ→1 AND γ→7/3
simultaneously — where the pre-declared rule already refuses a claim. Joint
Fisher quantification is a commissioning gate.

## Q1 — Factor of two in the §3 normalization.
**Real inconsistency, confirmed — the referee's most concrete catch.** The
derived covariance is Σ_sig = (S_arm/L²)·M(ρ) with diagonal 2S_h; the v0.3
"½ prefactor" sentence contradicted the variance equation two lines earlier.
Fixed: A_h is the **per-path** amplitude; per-channel signal PSD is
2A_h(f/f₀)^{-γ}. The code used Σ_sig = S_path·M everywhere (likelihood,
injections, Fisher, ULs), so **no numerical result changes**; the Planck
mapping is per-path and unchanged. Every quoted amplitude now states the
convention.

## Q2 — Asimov/noiseless injection.
**Run** (§Asimov test). Truth includes the likelihood's jitter guard (an
early version mixed evaluators and produced an impossible fit-below-warm-start
result — documented as a lesson in the revision record). Available deviance
matches Fisher at small A (12.5 vs 16 at 4σ_F). Λ_Asimov = 0.006/0.17/49/394/
1345 at 4/16/64/256/1024 σ_F: noiseless crossing ~31 σ_F, confirming the noisy
ladder; absorption 98–99.9% of available deviance — overwhelming, not total.

## Q6 — Cross-tunnel correlations.
**Derived, and the family is closed** (§Derivation): uniform cross-tunnel
correlation κ′ gives Σ_sig = (1−κ′)S_h·M(ρ_eff), ρ_eff = (κ−κ′)/(1−κ′).
Long-correlation-length disturbances appear at suppressed amplitude and
shifted ρ; at full coherence the triangle is blind (correct physical limit).

## Q7 — Low-frequency regularization / f_min sensitivity.
The f_min sweep (5→30 Hz) stands from the v0.3 log-bin run: LR conclusions
unchanged at all cutoffs, σ_F degrades only ×1.9 — the log binning removed
the pathological low-f dependence. A physical turnover below the band is
absorbed into the ⟨|T|²f^{-γ}⟩ bin averages.

## Q8 — Prove the fixed-A exclusion under profiling.
**Proven** (§Upper limit): Σ(θ;A) ⪰ A·s_k·λ_min[M(ρ)]·I = A·s_k(2−2ρ)·I in
Loewner order for every admissible nuisance (P⪰0 diagonal, BB† Gram, phases
unitary), so lnL_p(A) ≤ −3Σm_k ln[A s_k(2−2ρ)] → −∞. The three conditions
doing the work (ρ<1 fixed; factor enters as a Gram matrix; calibration
phase-only) are each enforced by the pipeline. Diagonals "moving down" cannot
escape: the bound never uses the fitted P.

## Q10 — Gauge fixing of b → e^{iθ}b.
§Fisher: the redundancy is one flat direction (constant θ; approximately flat
for smooth θ(f)); harmless in the optimizer (changes no likelihood value);
profiled Fisher uses the Moore–Penrose pseudo-inverse (rcond 1e-10) over the
nuisance block. Explicit gauges (Im b₁=0) rejected for their coordinate
singularity.

## Q11 — MDC1 channel conventions.
§Derivation: single-channel sign flips are absorbed exactly by the phase
splines (e^{iπ}), and t is invariant under per-channel sign flips outright
(each channel enters twice), so both channels are convention-robust.
Verifying the E1/E2/E3 ↔ cyclic-Michelson mapping stays a commissioning item
because the *interpretation* of a fitted φ ≈ π depends on it.

## Q12 — Sign invariant vs line forests.
Measured (line-forest MC, m=100): a strong **rank-1** line *suppresses* the
sample false rate of Re t<0 (0.02 vs 0.45 clean) — a single coherent source
cannot fake the sign, by the theorem. **Rank-2** (two independent sources in
one bin) opens the negative octant at the predicted ~17% (0.174 sample,
0.167 model). Masking + m^{3/2} weighting control it; the bootstrap
calibrates whatever remains.

## Q13 — Why does loud CBC content leave no signature?
§SGWB, three compounding reasons in measured order: time-frequency dilution
of a transient in a 128 s Welch average; a single CBC source is exactly
rank-1 in channel space, absorbed *legitimately* by the environmental factor;
and the residue sits below the sign channel's calibrated floor (coherences
at the 1/N_seg floor). The v0.2 "CBC cross-power" outlier story is doubly
dead.

## Code delivered with this round
- `qif_v2.py`: `loglike_et_qif_grad` (analytic scores), scaled optimization in
  `fit_model`, `sign_channel_stat`, `sign_channel_pvalue` (universal
  conservative calibration), `sample_wishart_bartlett` (exact O(1) sampler).
- `scripts/check_analytic_scores.py`: score-vs-FD regression test (PASS).
- `qif_v2_cuda.py`: flagged as FD-only pending a validated CuPy port.
