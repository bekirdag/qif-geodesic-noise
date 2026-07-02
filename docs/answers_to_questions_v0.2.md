# Answers to the referee questions on the v0.2 paper

**Date:** 2026-07-02 (same day as the v0.2 release; see the critical preamble below)
**Refers to:** `docs/et_geodesic_noise_paper_v0.2.pdf`, `docs/quesitons.md`
**New computations:** `test-runs/` and the re-run log referenced in each answer.

---

## Critical preamble: answering these questions exposed three further pipeline defects

While producing numerical answers to Questions 10 and 27, we found that the v0.2
"corrected" pipeline still contained a **fatal optimization defect**, different from the
two documented in the v0.2 post-mortem — and the subsequent re-validation exposed two
more problems (a nested-fit symmetrization gap and a data-discarding bin layout,
described after the main bug below). All three are fixed in the v0.3 implementation and
paper; all v0.2 numbers are superseded.

**The bug.** `scipy.optimize.minimize(..., method="L-BFGS-B")` approximates gradients by
finite differences with an **absolute** default step `eps ≈ 1.5e-8`. The environmental
factor coefficients `B` have physical scale `~1e-24` (square root of the strain PSD).
Perturbing `B` by `1.5e-8` inflates the model covariance `B Bᴴ` by ~32 orders of
magnitude, so the numerical "gradient" in those directions is `~1e24` — pure garbage.
Every line search failed, and L-BFGS-B exited at **iteration zero** (`nit=0`,
`status=2 ABNORMAL`) with the start point returned untouched.

**The smoking gun** (BBH_snr_306, 64 s / 64 bins): the "fitted" instrument PSD was
exactly the flat initialization (spline coefficient spread 0.00 across the band, data
dynamic range ×76), per-bin Wishart traces ranged 0.2–7.1 instead of ≈3, `max|grad| =
1.2e24`, and the achieved `lnL = 614760.8` versus `618725.6` for a perfect diagonal fit
— **~4,000 log-likelihood units left on the table**. All LR structure in v0.2 came from
the gradient-free 1-D profile scan over `log A_h` alone; the 240-dimensional joint fits
never moved. The tell-tale that led us there: the "n_coeff stability" test returned LR
values **byte-identical to 16 digits across spline bases of different dimension**, which
is impossible for working fits (identical statistics across different model dimensions is
the new diagnostic rule this adds to the v0.1 post-mortem list).

**The fix** (in `qif_v2.py`, mirrored in `qif_v2_cuda.py`):
1. per-parameter finite-difference steps matched to each block's scale
   (`eps = 1e-4` in log/logit/phase units; `eps = 1e-4 × B_scale` for B coefficients);
2. `maxfun` raised to accommodate finite-difference gradients (scipy's default 15,000
   silently truncates a 240-dimensional fit after ~60 gradient evaluations);
3. the P-splines are now initialized by least squares on the log periodogram instead of
   a flat median (the flat start is many e-folds from the answer).

After the fix the null fit reaches `lnL = 618685.8` (within 40 of the diagonal-perfect
bound, with smoothness constraints), the fitted PSD tracks the data across its full
dynamic range, and per-bin traces are ≈ 2.4–3.1 as they should be.

**Second defect (nested-pair symmetrization gap).** With working gradients, a single
cross-pollination round proved insufficient: the alternative's refit can enter a better
basin of the *shared* nuisance model after the null's last refit, producing a spurious
Λ = 19.8 with the fitted amplitude at the numerical floor (structurally incoherent —
if Â ≈ 0 the hypotheses coincide and Λ must be ≈ 0). Fixed by iterating
cross-pollination to convergence and bounding the final null value by a *direct*
(fit-free) evaluation of the null likelihood at the alternative's nuisance solution,
which caps Λ at twice the genuine amplitude contribution. New protocol rule: **a
positive Λ with Â at the boundary is a pipeline error**, symmetric to the negative-Λ
rule.

**Third defect (data-discarding bin layout).** The `max_bins` "downsampling" *subsampled*
every ~250th native Welch bin on a linear grid: 99.6% of the spectral data was discarded
and a single analysis bin sat below 100 Hz — starving a γ=2 search of exactly the
frequencies that carry its information (both the v0.2 "outlier" and the "8σ_F threshold"
lived in that one bin). Replaced by **log-spaced block averaging** of all native bins,
with the per-block effective Wishart count corrected for the Hann-window correlation
between neighboring native bins. With 64 log bins, 25 lie below 100 Hz, and the naive
Fisher scale improves ×4 while fits get *faster*.

**Consequences.** All v0.2 headline numbers (Table 1, bootstrap p-values, the injection
threshold, the profile upper limit) are superseded by the re-run below. The most
important scientific change: with a *working* nuisance fit, the flexible
instrument+environment model **absorbs most of the geodesic-diffusion template**, so the
v0.2 claim of an "8 σ_F effective threshold" was itself an artifact of frozen nuisance
fits — we verified the absorption is honest (the fitted signal-free model exceeds the
log-likelihood of the *generating truth* on injected data, so no better null optimum was
left unfound). The template's identifiable content against the rank-1 environmental
model reduces to a gauge-invariant sign obstruction (see Q6), and the honest sensitivity
is the *profiled* Fisher information (see Q26/the paper's forecast section). The paper
required a major revision, delivered as v0.3
(`docs/et_geodesic_noise_paper_v0.3.pdf`), itemized at the end of this document.

Re-run configuration (v0.3 protocol): Welch 4 s / 50% Hann, log-spaced averaged bins,
N_c = 20 log-knot splines, r = 1, fitted calibration phases, iterated guarded nested
pairs, profile-scan seeding, scale-matched finite-difference steps.

---

## Q1. Signal covariance derivation

**Answer: M(ρ) can be derived, including signs, the diagonal 2, and the ½ prefactor,
from a shared-tunnel path-length-diffusion model. It is less phenomenological than the
paper claims, and the derivation should be added.**

Model each interferometer output as the Michelson difference of the optical path noise
in its two arms, with a consistent cyclic orientation over the triangle:

    h₁ = (δa₁ − δb₁)/L,   h₂ = (δb₂ − δc₂)/L,   h₃ = (δc₃ − δa₃)/L,

where tunnel *a* hosts beams a₁ (detector 1) and a₃ (detector 3), etc. Assume each
beam's path noise has one-sided PSD S_arm(f), zero correlation between different
tunnels, and correlation κ(f) between the two beams co-located in the same tunnel.
Then per channel

    Var(hᵢ) = 2 S_arm/L²,     Cov(hᵢ, hⱼ) = −κ S_arm/L²   (the shared tunnel enters
                                                            once, with opposite signs).

Writing S_h = S_arm/L² for the per-channel strain PSD, the cross-spectral matrix is
exactly `Σ_sig = ½ S_h M(ρ)` with `ρ = κ`: the diagonal factor 2 is the two-arm
Michelson variance, the three negative off-diagonals are the shared-tunnel terms under
cyclic orientation, and the ½ prefactor is bookkeeping that makes `diag(Σ_sig) = S_h`
(so A_h is the per-channel strain PSD at f₀, which is what the pipeline fits).

So **ρ has a physical meaning: the transverse correlation coefficient of the diffusion
field between the two beams sharing a tunnel** (beam separation is meters; ρ → 0 for
microscopic correlation lengths, ρ → 1 for correlation lengths much larger than the
tunnel cross-section).

Structures intentionally (or implicitly) excluded by M(ρ):
- unequal per-tunnel correlations ρ₁₂ ≠ ρ₂₃ ≠ ρ₁₃ (three parameters instead of one);
- correlations between different tunnels (plausible if the diffusion field has
  km-scale coherence — precisely the ρ→1 regime, where the parameterization becomes
  degenerate anyway, see Q4);
- complex (delayed) correlations: light-travel-time phase factors matter above
  f ~ c/(2πL) ≈ 4.8 kHz; below ~1 kHz the corrections are O((2πfL/c)²) ≲ 4%, so the
  real-valued template is adequate in-band;
- unequal per-beam noise powers (absorbed to first order by the PSD splines).

Note the consistency check the derivation provides: at ρ = 1, M(1) is the graph
Laplacian of the triangle and the null-stream sum telescopes to zero — an arm-localized
but tunnel-coherent diffusion cancels in the null stream exactly like a GW (see Q4).

## Q2. Physical meaning of ρ(f)

Per Q1, ρ(f) is physical: the tunnel-scale transverse correlation of the diffusion
field, potentially frequency-dependent through the frequency dependence of the effective
sensing volume. The spline parameterization is a phenomenological proxy for that unknown
function, not pure nuisance flexibility.

On look-elsewhere: under H_env, ρ is unidentified (it multiplies A_h = 0) — a Davies
problem, so the ½χ²₀+½χ²₁ mixture is not exact (see Q14). The pipeline handles this
**empirically**: significance is calibrated by a configuration-matched parametric
bootstrap in which every replicate refits the same spline-ρ freedom, so whatever LR
inflation the extra freedom produces is present in the null distribution too. The
residual risk is misspecification of the bootstrap itself (Q15), not the ρ freedom per
se.

**Recommendation adopted for v0.3:** make constant-ρ (one parameter) the primary
analysis and spline-ρ a robustness variant. The physical prior from Q1 (a correlation
coefficient between two nearby beams sampling the same field) makes slow frequency
dependence plausible and 20 free coefficients gratuitous.

## Q3. A_h–ρ degeneracy

The decomposition M(ρ) = (2+ρ)I − ρJ (J the all-ones matrix) makes the structure
explicit: the isotropic part (2+ρ)A_h f^(−γ) is exactly degenerate with the PSD
splines; the identifiable combination is the off-diagonal product **β ≡ ρ·A_h**. A
ridge A_h ∝ 1/ρ at fixed β therefore exists by construction, terminated only by the
boundary ρ ≤ 1, which is what converts a measurement of β into the (weaker) bound
A_h ≥ β.

With the fixed optimizer this is no longer academic: the re-run shows the flexible
nuisance model absorbs essentially all of the diagonal information (see Q7/injections),
so the practical information content is β alone — and of β, only the part protected by
the sign obstruction of Q6.

**Recommendations for v0.3** (partially implemented in the re-run): report ρ̂ alongside
Â_h in the results table; quote β̂ = ρ̂·Â_h as the primary fitted quantity and upper
limits on β; show the joint (A_h, ρ) profile in the paper. The Fisher forecast should
be restated for β at fixed ρ (its current form, with ρ fixed at 0.5, effectively is
that already, but the text should say so).

## Q4. The ρ → 1 null-stream limit

No physical prior excludes ρ → 1. Per Q1, ρ → 1 corresponds to a diffusion field
coherent across the tunnel cross-section (correlation length ≫ beam separation of
meters) — for any "spacetime foam" motivation with microscopic correlation lengths one
expects ρ ≈ 0, but the metric-level (long-wavelength) reading gives ρ ≈ 1, and the
search cannot legislate between them.

Consequences, stated honestly:
- at ρ → 1 the null stream retains signal power ∝ (1−ρ) → 0, so the designated
  discriminator against GW backgrounds fades exactly where the signal is most
  GW-like;
- discrimination against a stochastic GW background then rests entirely on spectral
  shape (f^(−2) vs. the CBC background's ≈ f^(−7/3) — uncomfortably close over one
  decade) and on the rank structure (Q23): an isotropic SGWB in a closed triangle has a
  rank-2 channel covariance (the null-stream eigenvector has eigenvalue zero), while
  M(ρ<1) is rank 3. At ρ = 1 exactly, M(1) is also rank 2 and the degeneracy with an
  SGWB of the same spectrum is **complete** at the covariance level.

**Recommendation for v0.3:** state explicitly that the search targets ρ < 1; quote
results at fixed ρ ∈ {0.25, 0.5, 0.75} as the primary grid; and treat any candidate
with ρ̂ → 1 as indistinguishable from a GW background pending a shape/isotropy
analysis.

## Q5. Full ET response bias

The requested injection test (inject with realistic R(f)C_ℓ(f)R†(f), recover with
M(ρ), report bias) has not been run; it stays on the commissioning list and belongs
there. But the question deserves a sharper answer than the paper's blanket caveat,
because the size of the R ≈ I error depends on the *interpretation* of the signal:

- **Path-length-noise reading** (the reading actually used by the Planck mapping,
  Eq. 2): the disturbance adds to the measured optical path directly. There is no
  antenna pattern to project; R = I is *exact* up to light-travel-time (delay)
  corrections of O((2πfL/c)²) — about 0.04% at 100 Hz and ~4% at 1 kHz for L = 10 km —
  and up to the sensing/calibration response, which the pipeline already accepts as an
  external transfer function. For this reading the v0.2 caveat overstates the problem.
- **Metric/strain reading** (and for any GW background in the data): the 60° arms,
  polarization projections, and frequency-dependent response reshape the cross-channel
  structure at O(1), and fitted (ρ, A_h) inherit those errors. This is the channel for
  which "R ≈ I is not benign" is correct.

**Recommendation for v0.3:** split the response discussion by interpretation as above;
keep the full-response injection test as the commissioning gate for the strain reading.

## Q6. Gauge/sign invariance of the off-diagonal argument

**The identifiability statement is exactly gauge-invariant, and we can now state it as
a small theorem** (numerically verified; see the session log):

For a Hermitian 3×3 covariance Σ define the triple product t = Σ₁₂Σ₂₃Σ₃₁.
- Under per-channel phase calibration G = diag(e^{iφᵢ}) (and any channel sign flips),
  t is invariant: the phases cancel around the closed loop 1→2→3→1.
- For any model of the form D + b bᴴ (diagonal + rank-1, arbitrary complex b, with or
  without phase calibration): t = |b₁b₂b₃|² ≥ 0, real. Adding any diagonal never
  changes the off-diagonals.
- For the signal template: t = (−ρ A_h S̃(f))³ < 0.

So "three negative real off-diagonals" is shorthand for the invariant statement
**Re t < 0, which no diagonal + rank-1 + phase-calibration model can produce at any
frequency**. Verified numerically: t real ≥ 0 for random rank-1 draws to machine
precision, exactly invariant under random phase rotations.

**The limitation the paper must add:** the theorem dies at rank 2. Random complex 3×2
factors produce Re t < 0 in ~17% of draws, and — more fundamentally — by the Ledermann
bound, `diagonal + rank-2` can reproduce **any** 3×3 Hermitian covariance exactly
(three channels support at most one identifiable factor). Against an r = 2
environmental model the signal is identifiable only through frequency structure
(smooth splines vs. the f^(−γ) power law), which is weak. See Q7.

## Q7. Rank-2 environmental stress test

*(Numbers from the fixed-optimizer re-run; the pre-fix r = 2 results were vacuous —
the r = 2 fits never moved, which is why v0.2 could only "mention" the stress test.)*

With the v0.3 pipeline (log grid, guarded pairing, 64 s / 64 bins):

| configuration | r = 1 | r = 2 |
|---|---|---|
| BBH_snr_306, real data | Λ = 0.000 | Λ = 0.000 |
| BNS "outlier" segment, real data | Λ = 0.000 | Λ = 0.000 |
| 64 σ_F injection (A = 1.1e-46) | Λ = 0.000 | Λ = 0.000 |

The r = 1 vs r = 2 comparison is moot in the tested regime: the *rank-1* model
already absorbs the template completely (see Q16), so rank 2 has nothing left to
absorb. The question's concern is nonetheless vindicated theoretically — see the
Ledermann-bound discussion below and in the paper §5: r = 2 + diagonal can represent
*any* 3×3 Hermitian covariance, so it is structurally a GW-contamination veto, not a
null test.

Interpretation, combining with Q6:
- Per the Ledermann bound, r = 2 + diagonal is an over-complete per-frequency model:
  it can absorb *any* cross-spectral structure, including the geodesic template, up to
  the spline smoothness constraint. A "detection" that survives r = 1 but vanishes at
  r = 2 is therefore expected even for a true signal — **the r = 2 fit is not a valid
  null test for the geodesic hypothesis; it is a GW/CBC contamination veto** (a GW
  background occupies a rank-2 subspace and is fully absorbable, Q23).
- The paper must reframe the "stress test" accordingly: r = 1 with the sign obstruction
  is the discovery configuration; r = 2 is diagnostic only.

## Q8. Calibration nuisance dimensionality

The φ's are **not** per-bin parameters. `phi_coeffs` is an (N_c × 2) array: two smooth
open-uniform cubic B-spline curves over the analysis band (channels 2 and 3; channel 1
gauge-fixed), i.e. 40 calibration dof at N_c = 20 — not 2 × 128 = 256 per-bin dof. The
paper's notation φ_{2k}, φ_{3k} means "the spline evaluated at bin k" and should say
so explicitly (v0.3 edit).

The requested ablation (per-bin vs. spline vs. fixed) has not been run; the
`--calib-variants` driver already implements fixed-vs-fitted and belongs in the
commissioning suite. Per-bin phases would be a mistake: by the counting of Q6/Q7 they
hand the null model enough freedom to rotate off-diagonal structure bin by bin, and
they would destroy the phase-coherence part of the identifiability argument.

## Q9. Missing amplitude calibration uncertainty

Correct observation; the answer has two parts.

*Where amplitude errors go now.* A per-channel amplitude error gᵢ(f) multiplies
Σ → diag(g) Σ diag(g). Its effect on the diagonal is absorbed exactly by the PSD
splines (they are free log-amplitude curves). Its effect on the off-diagonals
multiplies Σᵢⱼ by gᵢgⱼ > 0: it rescales the identifiable combination β = ρA_h by
roughly (1 + εᵢ + εⱼ) for fractional errors ε, but — crucially — it is a *positive*
rescaling: it cannot create or destroy the sign obstruction of Q6. So amplitude
calibration errors bias |Â_h| at O(ε) (percent-level for percent-level calibration)
but do not generate false positives through the sign channel.

*What should change.* For production, add smooth per-channel log-gain nuisances with
tight priors (measured calibration uncertainty), which formalizes the above instead of
leaving it implicit in the PSD splines. This is a small code change and a v0.3
commissioning-plan item.

## Q10. Spline basis versus broadband signal absorption

This question is what exposed the eps bug (see preamble): the v0.2-era n_coeff
stability check returned byte-identical LR across N_c ∈ {20, 28, 36}, which is
impossible for working fits and turned out to mean "the fits never moved."

With the fixed optimizer, on a loud (64 σ_F) injection:

| N_c | Λ | Â_h | lnL_env |
|---|---|---|---|
| 20 | 0.000 | 8.7e-60 | 81420159.07 |
| 28 | 0.000 | 8.7e-60 | 81419977.85 |
| 36 | 0.000 | 8.7e-60 | 81420187.52 |

(64 σ_F injection, 64 s / 64 log bins.) The lnL values now *differ* across bases —
the fits are alive (contrast the byte-identical v0.2 values) — while Λ is stably zero
because the absorption is basis-independent. Note the lnL differences across N_c
(~200 units) also measure the per-fit convergence scatter at this likelihood scale;
see Q26/Q27.

The structural answer to "what prevents too many coefficients from subtracting a
genuine signal":
- the *diagonal* part of a genuine f^(−2) signal is **not** protected — any
  sufficiently rich smooth diagonal model absorbs it, at any N_c (measured: see the
  injection ladder in Q16/Q7 — this is now the dominant sensitivity penalty, not a
  hypothetical);
- the *off-diagonal sign obstruction* (Q6) is protected against the diagonal + rank-1 +
  phases model at any spline richness, because no member of that family can produce
  Re t < 0 at any frequency;
- therefore LR stability under N_c is a necessary but weak check; the primary
  protection is structural (the sign channel), not basis-size tuning.

Knot placement: the knots are currently uniform in *linear* frequency, which places
almost no resolution in the decade below ~400 Hz where a γ = 2 signal lives, and none
between 5–10 Hz. The f_min = 5 Hz entry of the cutoff sweep (Q21) shows the
consequence. **v0.3 change: uniform-in-log-f knots** (one-line change in
`_open_uniform_knots` usage), plus an explicit knot-placement robustness variant.

## Q11. High-Q line leakage

Not yet addressed empirically, and the MDC1 frames cannot address it (their noise is
smooth colored Gaussian, no line forest). Status: open; commissioning item. What can be
said structurally:

- instrumental lines are per-channel (diagonal): they cannot forge the off-diagonal
  sign pattern; their damage is misfit noise in both hypotheses (inflated LR variance)
  and PSD-spline distortion that leaks into the diagonal-absorption budget;
- correlated lines (mains harmonics, common clocks) are the dangerous class: they are
  genuine cross-channel power. They are narrow, so the f^(−2) template gains little
  from them per line, but a dense forest could accumulate. Masks (`--line-mask`,
  already implemented) plus witness-channel vetoes are the mitigation;
- Hann + 50% overlap gives −31.5 dB first sidelobe and rapid rolloff; the residual
  leakage wings after masking ±few bins are far below the broadband target, but this
  must be *demonstrated* with a synthetic line forest before real-data claims — a
  cheap, fully specified simulation that belongs in the v0.3 validation suite.

## Q12. Frequency-bin correlations

Correct: with Hann windows and 50% overlap, neighboring native FFT bins are correlated,
and the analysis-level bins (block averages of native bins) inherit reduced but nonzero
edge correlations; the Wishart likelihood ignores this, and — importantly — **the
parametric bootstrap inherits the same independence assumption**, so it does not
calibrate this error away. The effect has not been measured; expected direction: mild
overdispersion of Λ relative to the bootstrap null, i.e. slightly optimistic p-values.

Mitigations, in order of rigor: (i) time-domain end-to-end simulation (Q13) — the
definitive measurement; (ii) decimate analysis bins (keep every second block) as a
robustness variant; (iii) inflate m_eff by the measured inter-bin correlation. Status:
open; the time-domain validation is the designated closure and is fully specified in
Q13.

## Q13. Welch-overlap Wishart approximation

Agreed, and the validation is designed (not yet run): generate Gaussian time series
with a known Σ(f) by frequency-domain factorization; push them through the *identical*
Welch pipeline (4 s Hann, 50% overlap, same bin averaging); fit the nested pair with
the full protocol; repeat O(10³) times; compare the empirical Λ distribution against
(a) the ½χ²₀+½χ²₁ mixture and (b) the parametric bootstrap distribution. Runtime at
the 64-bin configuration is ~20 s per replicate on this laptop (measured), so 10³
replicates ≈ 6 CPU-hours — feasible without a cluster. This closes Q12 and Q13
simultaneously and is the top-priority v0.3 validation item.

## Q14. Bootstrap validity near a boundary

Correct on both counts. (i) At A_h = 0, ρ (and with spline-ρ, its whole curve) is
unidentified — a Davies problem stacked on the boundary problem, so the asymptotic
mixture is at best an approximation; the pipeline's significance statements rely on the
configuration-matched bootstrap, not on asymptotics (the mixture appears in the paper
only as qualitative context for the boundary mass). (ii) Convergence in n_b: current
runs (n_b = 20–40) resolve p only down to ~0.03–0.05 and say nothing about the
detection-relevant tail. A pre-declared detection threshold needs n_b ≥ 10³–10⁴
replicates at matched configuration (cost: Q26), plus a tail fit (e.g. generalized
Pareto) with its own uncertainty budget. v0.3 should state this explicitly and refrain
from quoting any p below ~1/n_b.

## Q15. Parametric bootstrap under model misspecification

Correct: the parametric bootstrap answers "what does Λ do under *my* null model with
*my* optimizer," not "under the actual noise." Planned comparison set for v0.3
commissioning, in increasing order of realism: (i) parametric Wishart bootstrap
(current); (ii) segment-level block bootstrap — resample the Welch segment index before
averaging, preserving per-segment non-Gaussian structure at near-zero implementation
cost; (iii) time-domain Gaussian resimulation (Q13); (iv) real signal-free stretches
(Q18). Discrepancies between (i) and (ii)–(iv) directly measure the misspecification
penalty on false-alarm calibration. Only (i) exists today; that is stated as a
limitation in the paper and remains one.

## Q16. Time-domain injection recovery

Not yet done; the current injections are Wishart-level resamples conditioned on the
fitted environmental model, which validate the likelihood/optimizer machinery only.
(With the optimizer fixed, even these changed qualitatively — see the re-run ladder
below.) Frame-level injections through the full pipeline (windowing, overlap, leakage,
line masks, non-stationarity, CBC subtraction residuals) with detection-efficiency
curves are a commissioning-gate deliverable and are straightforward given Q13's
simulation machinery.

Fixed-optimizer Wishart-level ladder (BBH_snr_306-conditioned, 64 s / 64 bins,
ρ_inj = 0.5), which now *replaces* the v0.2 Fig. 4 and its "8 σ_F threshold":

| A_inj/σ_F | A_inj [Hz⁻¹] | Λ | Â_h [Hz⁻¹] |
|---|---|---|---|
| 0 | 0 | 0.000 | 7.3e-60 |
| 4 | 7.1e-48 | 0.000 | 7.4e-60 |
| 8 | 1.4e-47 | 0.000 | 7.5e-60 |
| 16 | 2.8e-47 | 0.000 | 7.6e-60 |
| 32 | 5.7e-47 | 0.000 | 8.3e-60 |
| 64 | 1.1e-46 | 0.000 | 8.6e-60 |
| 128 | 2.3e-46 | 0.000 | 1.0e-59 |
| 256 | 4.6e-46 | 0.000 | 1.2e-59 |

(σ_F = 1.78e-48 Hz⁻¹, naive Fisher at the fitted null, log grid.)

**No injection is recovered at any tested amplitude** — the unconstrained
diagonal+rank-1+phases nuisance model absorbs the template completely up to
256 σ_F = 4.6e-46 Hz⁻¹. We verified the absorption is honest: the fitted
*signal-free* model exceeds the log-likelihood of the generating truth (by +29 at
256 σ_F — normal Wishart overfitting), so no better null optimum was left unfound.
Mechanism (numerically verified): the rank-1 tangent space covers 5 of the 6
off-diagonal degrees of freedom, and the manifold's nonlinear freedom to relocate the
fitted factor shadows the template in the remaining direction — the sign obstruction
of Q6 is a *curvature* (higher-order) effect that the profile LR does not feel at
these amplitudes. Consequence: the LR search as specified is an upper-limit machine,
not a detection machine, until the environmental model is constrained by external
information (witness channels, measured couplings, clean-stretch solutions) or a
dedicated sign-channel statistic is added. This finding, not any threshold number, is
the honest answer to the question.

## Q17. BNS outlier null-stream decomposition

Three independent data-level diagnostics on BNS_snr_379 gps=1001329152 (the v0.2
Λ = 51.5 outlier), none of which involve the optimizer:

- **Null-stream ratio** (128 s, Welch 4 s/50%): (E₁+E₂+E₃) ASD / single-channel ASD =
  1.7315 broadband (10–1024 Hz) and 1.7241 in 10–128 Hz, vs. √3 = 1.7321 expected for
  independent noise — identical to the two quiet BNS segments (1.7284/1.7267 and
  1.7290/1.7388). No median-level excess or deficit.
- **Pairwise coherences** (10–128 Hz medians): 0.011–0.012 in all three pairs, at the
  1/N_seg floor — again identical to the quiet segments.
- **Off-diagonal sign pattern** (median Re Sᵢⱼ/P̄, 10–128 Hz): (−4.5, +1.1, −13.4)×10⁻³
  for the outlier vs. (−5.2, +10.4, −2.2)×10⁻³ and (+8.6, −1.4, +7.3)×10⁻³ for the
  quiet segments — random signs at noise level, **not** the coherent all-negative
  pattern a geodesic-diffusion component would imprint.

So at the level of stationary spectral statistics the "outlier" segment is
indistinguishable from its quiet neighbors: a time-localized CBC contributes little to
128-s medians. And with the fixed optimizer the fitted outlier itself deflates:

with the v0.3 pipeline the outlier's LR collapses from 51.52 to **0.000**
(128 s / 128 log bins) — the entire excess was nuisance-model misfit that the crippled
optimizer could not remove, concentrated in the single under-resolved low-frequency
bin of the old subsampled linear grid.

The v0.2 narrative ("Λ = 51.5 = real correlated CBC cross-power partially fit by the
template") was therefore wrong in an instructive way: the excess was concentrated in
the lowest analysis bin and was largely *nuisance misfit* that the crippled optimizer
could not remove; the honest attribution is optimizer failure first, CBC content
second. v0.3 must retract that interpretation.

## Q18. Clean-noise or CBC-subtracted validation

Not yet available, and the answer should be blunt: the mirrored "loudest" archive
contains only CBC-loud segments, so **no statement about production false-alarm
behavior can be made from the data currently in the repository**. The paper says this,
and the re-run reinforces it (every fitted group is now consistent with the bootstrap
null, so the shakedown carries no false-alarm information beyond n=8 segments). The
concrete v0.3 action: fetch signal-free (or sparse-injection) stretches from the ET
MDC origin (the full MDC1 dataset includes month-scale streams beyond the "loudest"
subsets) and produce the requested validation table; nothing in the pipeline blocks
this beyond download volume.

## Q19. Robustness of the null stream

For the equal-response ideal, u = (1,1,1)/√3 nulls GWs exactly. With per-channel
response/calibration errors εᵢ(f), the optimal null combination generalizes to the
minimum-variance GW-orthogonal projector w(f) built from the calibrated responses; for
small errors, using the naive u instead leaks GW power ~ |ε|² (power leakage is
second-order because the leakage amplitude is first-order and u is a stationary point
of the GW response on the simplex). A 10% calibration error therefore leaves a ≤ 1%
GW-power leakage floor — adequate for the veto role the null stream plays here, though
not for precision null-stream *measurement* of (1−ρ)S_h. The geodesic content of the
null stream, uᵀM(ρ)u = 2(1−ρ), degrades gracefully under the same errors. A
quantitative leakage curve with realistic ET calibration budgets is a commissioning
item; the qualitative conclusion (null stream survives as a veto at realistic
calibration accuracy, except in the ρ→1 limit of Q4) is safe.

## Q20. Coverage of the profile-likelihood upper limit

Untested, and the criticism lands harder post-fix: the re-run UL is

A_h(95%) ≲ 7.7e-48 Hz⁻¹ (BBH_snr_306, 128 s / 128 log bins, template ρ = 0.5,
conservative). The limit is *optimizer-noise-limited, not statistics-limited*: at
lnL ~ 1.7e8 the finite-difference gradient noise floor is ~0.4/component and
independently-restarted fits scatter across nuisance basins by O(50–80) lnL units
(measured: +76.4), so the textbook Δ=2.71 crossing (~2e-48) cannot be certified. The
quoted limit uses a threshold inflated by 2× the measured scatter, referenced to the
best null found anywhere. Analytic Wishart gradients (closed form) remove the
limitation. Note the structural asymmetry that keeps ULs meaningful despite zero
detection power: nuisances can *imitate* an unmodeled signal but cannot *cancel* a
fixed positive-semidefinite signal term (P > 0, BB† ⪰ 0), so exclusion works where
detection does not.

Because the identifiable quantity is β = ρA_h (Q3), a profile UL on A_h alone leans on
the ρ ≤ 1 boundary, and its frequentist coverage under realistic (non-Gaussian,
drifting) noise is unknown. Planned v0.3 treatment: (i) quote the UL on β as primary;
(ii) run an injection-based coverage study (inject known A, compute UL, tabulate
empirical coverage vs. nominal 95% over ≥ 500 replicates — ~3 CPU-hours at the 64-bin
configuration); (iii) repeat under the block bootstrap of Q15 to probe non-Gaussian
robustness. Until then the UL is quoted as "procedure-defined," not
"coverage-verified."

## Q21. Low-frequency cutoff dependence

Fixed-optimizer sweep on BBH_snr_306 (64 s / 64 bins), with the Fisher scale σ_F
recomputed from the fitted null at each cutoff:

| f_min [Hz] | Λ | σ_F (naive) [Hz⁻¹] |
|---|---|---|
| 5 | 0.000 | 1.5e-48 |
| 10 | 0.000 | 1.8e-48 |
| 15 | 0.000 | 2.1e-48 |
| 20 | 0.000 | 2.3e-48 |
| 30 | 0.000 | 2.9e-48 |

(BBH_snr_306, 64 s / 64 log bins, v0.3 pipeline.) Two changes vs the broken run: the
spurious f_min = 5 Hz excess (LR ≈ 1035 with the old pipeline, an artifact of the
band-edge + linear-knot misfit) is gone, and the cutoff dependence of σ_F is far
milder (×1.9 from 5→30 Hz instead of ×35) because log-spaced bins resolve the whole
low-frequency decade instead of hanging the sensitivity on one bin. The forecast
still degrades toward higher cutoffs, so quoting reach as a function of f_min remains
the right presentation.

Structural conclusions:
- σ_F degrades steeply with the cutoff (γ = 2 concentrates information at the lowest
  usable frequencies): the forecast's reach is a statement about the 10–30 Hz decade,
  exactly the band where seismic/Newtonian/suspension systematics live. v0.3 should
  present the forecast as a function of f_min, not a single number.
- The f_min = 5 Hz entry is a warning label, not a sensitivity gain: below 10 Hz the
  MDC1 synthesis band edge produces a steep non-physical feature, and the
  uniform-in-linear-f knots (Q10) have essentially no resolution there, so the fit is
  dominated by unmodelable structure. Production choice: log-f knots and a cutoff at
  or above the instrument's validated band edge.

## Q22. Long-term T_obs^(−1/2) forecast

Intended combination: a single shared A_h (more precisely β) across segments, with
per-segment PSD/environment/calibration nuisances — the joint lnL is the sum of
per-segment lnL's and the segment-level Fisher information adds, giving T^(−1/2) as
long as per-segment systematic biases average down. The scaling breaks when a
*common-mode* bias b (shared calibration error, response-model error, correlated
environmental noise such as Schumann-band magnetics, unsubtracted CBC foreground)
stops averaging: the limit is max(σ_stat(T), b). With the identifiable channel being
the off-diagonal sign structure, the most plausible systematic floors are correlated
magnetics and the CBC foreground (Q23), not per-channel PSD drift (which is absorbed
segment-by-segment). v0.3 additions: state the shared-β hierarchical model explicitly;
add split-half consistency (β̂ in disjoint epochs) and a segment-regression drift
monitor as the standard diagnostics; present the forecast with an explicit systematic
floor parameter.

## Q23. Astrophysical foreground confusion

The sharpest version of this concern, now stated with the rank machinery:

- An isotropic SGWB in a closed triangle produces a channel covariance S_gw(f)·Γ where
  Γ (the ORF matrix) annihilates the null-stream direction: **rank 2**. The geodesic
  template M(ρ<1) is rank 3. Consequences: (i) an r = 2 environmental factor can
  absorb an SGWB *exactly* (Q7's veto role); (ii) a rank test — or simply the null
  stream — separates the two signal classes in principle.
- The danger zone: the CBC background spectrum S_h ∝ f^(−7/3) is close to the f^(−2)
  template over the sensitive decade, so under r = 1 with ρ → 1 (where M also
  degenerates toward rank 2 and the null stream empties) an unsubtracted CBC
  foreground is a near-perfect geodesic impostor. At ET sensitivity the CBC background
  is *loud*; a production search cannot treat it as negligible.
- Therefore: a joint model is required, not optional — fit S_gw·Γ (rank-2, known ORF)
  and β·(f/f₀)^(−2)·[M(ρ) structure] simultaneously; the geodesic detection statistic
  must be the improvement over the *foreground-inclusive* null. v0.3 must add this to
  the commissioning plan as a blocking item for any real-data claim.

## Q24. Existing-bound comparison

The requested table is necessary and, when built honestly, it **changes one of the
paper's headline claims**. The mapping rules:

- *Strain reading* (A_h is an instrument-independent metric-noise PSD): map each
  experiment's strain-PSD bound at its own frequency via A_h = S(f)·(f/f₀)². This is
  the reading behind the paper's numbers (resonators → 3.6e−29 and 2.5e−33 Hz⁻¹;
  TAMA-class → 2e−35 Hz⁻¹; ET forecast 12–16 orders below).
- *Path-length reading* (the reading the Planck mapping Eq. 2 actually uses: an
  L-independent path-noise PSD S_δℓ, with S_h = S_δℓ/L²): bounds transfer between
  instruments as A_h^(ET) = A_h^(exp)·(L_exp/L_ET)². Short-baseline experiments are
  then *more* sensitive to the underlying disturbance, because fixed path noise is a
  larger strain on a shorter baseline. An optical resonator with L_exp ~ 0.1 m and a
  strain-level bound of 2.5e−33 Hz⁻¹ maps to α ≲ 10⁻⁷ in Eq. 2's own normalization —
  i.e. **under the paper's own Planck mapping, the α = 1 random-walk benchmark is
  plausibly excluded by existing short-baseline experiments by orders of magnitude**,
  and the "α = 1 is currently allowed" claim holds only under the strain reading.
- The same L² issue invalidates the v0.2 sentence "if it is a path-length diffusion
  with L-independent S_δℓ … longer arms win": for detecting L-independent path noise,
  longer arms *lose* (the strain signal scales as 1/L²) — the sentence is backwards
  as written.

v0.3 must: (i) include the full table (experiment, measured quantity, frequency, PSD
convention, baseline, both mapped values); (ii) recompute the α reach under both
readings; (iii) fix the "longer arms win" sentence; (iv) restate the motivation
honestly — ET's case is strongest under the strain/metric reading and must be argued
there, or the L-scaling of the underlying model must be treated as a parameter to be
constrained jointly by short- and long-baseline experiments (which is itself an
interesting reframing).

## Q25. Pre-declared detection rule and trials factor

Proposed pre-registration for the production search (v0.3 should contain a version of
this verbatim):

1. **Primary configuration, fixed in advance:** γ = 2; constant ρ profiled over [0.05,
   0.95]; N_c = 20 log-f knots; r = 1; smooth-spline phases; declared band; declared
   line masks and segment-selection rules.
2. **Trials-controlled secondary grid:** γ ∈ {1, 2, 3, 4}, three fixed-ρ values —
   Bonferroni over the declared grid; everything else (knots, N_c, bootstrap mode,
   masks) is a *robustness* axis whose variations may only veto, never promote, a
   candidate.
3. **Detection requires all of:** (a) config-matched bootstrap p < 10⁻³ with n_b ≥ 10⁴
   (or time-domain-simulation calibration at that level); (b) β̂ consistent across
   independent segment subsets (split-half); (c) null-stream power consistent with
   (1−ρ̂)·S_h(f) — with the ρ→1 caveat of Q4 declared as "GW-indistinguishable";
   (d) the off-diagonal sign pattern (Re t < 0) present in the raw binned
   cross-spectra, not only in the fit; (e) injection efficiency > 90% demonstrated at
   β̂ under the primary configuration; (f) survival of the foreground-inclusive joint
   fit of Q23; (g) environmental/witness-channel vetoes clean.
4. Any post-hoc configuration change reclassifies the result as exploratory.

## Q26. Computational scaling

Measured on this laptop CPU (vectorized batched-Cholesky likelihood, fixed optimizer,
finite-difference gradients over ~240 parameters): one nested pair ≈ 15–20 s at 64
bins; ≈ 50–65 s per group at the 128-bin table configuration (~73 s per group at the 128-log-bin table configuration with the iterated guarded pairing).
Cost driver: finite differencing (dim+1 likelihood evaluations per gradient).

Projections: n_b = 10⁴ full-refit bootstrap replicates at the 128-bin configuration ≈
10⁴ × ~1 min ≈ 170 CPU-hours — trivial for a cluster (embarrassingly parallel) and
overnight for one modern multi-core node; r = 2 roughly doubles the parameter count
and ~2–3× the cost. The honest large-scale plan: (i) analytic gradients (the Wishart
score is closed-form; removes the (dim+1) factor — an order of magnitude); (ii) GPU
batching of replicates (the CUDA path already vectorizes the likelihood); (iii)
generalized-Pareto tail modeling to reach effective p ~ 10⁻⁵ from 10⁴ replicates, with
the tail-fit uncertainty propagated; (iv) warm-starting replicate fits at the
generating parameters (valid — it only makes the null fit better, which is
conservative for p).

## Q27. Optimizer diagnostics

The question is answered by events: applying exactly the requested diagnostics to the
v0.2 fits (gradient norms, `nit`, boundary hits, stability under model-dimension
changes) is what exposed the eps bug within the hour. Concretely measured on a v0.2
fit: `nit = 0`, `status = 2 (ABNORMAL)`, `max|∇| = 1.2e24`, 0 of 240 parameters moved,
lnL 3,965 units below the diagonal-fit bound. After the fix: `nit` in the hundreds,
per-bin Wishart traces 2.4–3.1, PSD splines tracking the data's full dynamic range.

For v0.3, per-fit diagnostics become part of the reported results, not an internal
matter — every headline number carries: optimizer status and iteration count;
max-scaled-gradient at the optimum; count of active (boundary) parameters;
multi-start dispersion of lnL; whether cross-pollination improved the null (it should
rarely win — frequent wins indicate asymmetric optimization); and a
**dimension-invariance check** (results byte-identical across different N_c or r are
a bug signature, the inverse of the usual stability check). Cross-backend agreement
(CPU vs. CUDA) remains unverified — no NVIDIA GPU is available on the development
machine; the CUDA path mirrors every fix but has never executed. That fact belongs in
the paper's implementation section.

---

# Required revisions to the paper (v0.2 → v0.3)

**Blocking (results are wrong or superseded):**
1. **New post-mortem + full re-run.** Add the finite-difference eps bug as a third
   post-mortem entry (absolute FD steps vs. 1e−24-scale parameters; nit=0 ABNORMAL;
   byte-identical results across model dimensions as the new diagnostic signature).
   Replace *all* empirical numbers: Table 1, bootstrap, injections, UL, and the
   figures built from them.
2. **Retract the "8 σ_F effective threshold" and the identifiability-penalty
   narrative** (§8.3/§4 of v0.2): with working nuisance fits the flexible model
   absorbs the template's diagonal entirely; the identifiable content is the
   off-diagonal sign obstruction, and the measured threshold is
   not reached anywhere below 256 σ_F = 4.6e-46 Hz⁻¹ at 64 s — the unconstrained search has no measurable detection power in the tested range. The Fisher-to-threshold "gap" discussion must be
   rewritten around β = ρA_h.
3. **Retract the BNS-outlier interpretation** (Λ = 51.5 as CBC cross-power): the
   fixed-optimizer value is 0.000 (128 s / 128 log bins, v0.3 pipeline), and the data-level diagnostics (null
   stream, coherences, sign pattern) show no stationary correlated excess.
4. **Fix the existing-bounds comparison (Q24):** the "α = 1 currently allowed" claim
   conflates the strain and path-length readings; under the paper's own Eq. 2 mapping,
   short-baseline bounds on α are ~orders stronger. Add the two-reading table, fix the
   backwards "longer arms win" sentence, restate the motivation under the reading
   where it actually holds.

**Substantive additions:**
5. Add the M(ρ) derivation from the shared-tunnel model (Q1) — it upgrades the
   template from ad hoc to physically parameterized, gives ρ a meaning, and yields the
   ρ = 1/null-stream consistency check.
6. State the identifiability theorem and its rank-2 limitation (Q6/Q7): triple-product
   invariance; Ledermann bound; r = 2 reframed as a GW-contamination veto, not a null
   test.
7. Add the SGWB confusion analysis (Q23): rank-2 ORF structure, f^(−7/3) vs f^(−2)
   proximity, joint-foreground fit as a blocking commissioning item.
8. Report ρ̂ (or β̂ = ρ̂Â_h) in the results table; quote limits on β as primary (Q3).
9. Pre-declared detection rule (Q25) as a numbered subsection.
10. Optimizer diagnostics table for every headline result (Q27); note the CUDA path is
    untested on the development machine.

**Smaller edits:** log-f knots (Q10/Q21); forecast as a function of f_min (Q21);
φ-spline dof stated explicitly (Q8); amplitude-calibration nuisance plan (Q9);
response discussion split by signal interpretation (Q5); ρ→1 caveat and fixed-ρ
primary grid (Q2/Q4); shared-β hierarchical model and systematic floor in the 1-yr
forecast (Q22); bootstrap n_b / tail-resolution caveat (Q14); block bootstrap in the
calibration comparison plan (Q15).
