# Answers to the fourth-round (v0.3.2) referee questions → v0.3.3

Resolved in `docs/et_geodesic_noise_paper_v0.3.pdf` (v0.3.3, 2026-07-05).
Numbers: `test-runs/rerun_results_v033.json`. Referee A = three edge-case
questions; Referee B = fourteen numbered questions. One genuine pipeline
improvement came out of this round (see A1/B-seeding below).

## A1 — Frequency-independent ρ vs a physical κ(f)?
The premise about the physics is right; the pipeline concern dissolves once
stated precisely: the *fitted* ρ(f) is an N_c-coefficient spline in every
likelihood fit — "constant ρ" constrains only the trials-controlled template
grid and fixed-ρ ULs. **But measuring it exposed a real seeding weakness**:
with the 1-D amplitude scan (ρ pinned at its flat init), decaying-κ(f)
injections were found in only 2/6 trials. Fixed: the seeding now scans
(amplitude, constant ρ) jointly (three ρ configurations); efficiency recovers
to 4/6 vs 13/16 for the matched template (small-sample consistent), amplitude
unbiased on detection (Â/A = 0.80–1.04), fitted ρ(f) tracks the injected
decay. Mismatch against a constant-ρ *template* deflates recovery rather than
biasing it up — conservative in the direction that matters (β = ρ(f)A_h per
bin). Paper §Derivation + protocol item 4.

## A2 — Cross-block Hann leakage vs Wishart independence?
Bounded analytically and numerically: only edge native bins couple adjacent
blocks, so block-average correlation ≲ ϱ₁²/(n_i n_j). Measured: worst
adjacent-block correlation 0.028 on the 64-bin grid (min n = 4), 0.11 on the
128-bin grid (min n = 2), falling as 1/n² toward high frequency — the log
grid is *widest in native-bin count* exactly where it is narrowest in
log-width, killing the high-frequency version of the concern. Percent-level
information overstatement in the lowest blocks; m_eff de-rating for n<4
blocks available; time-domain closure still the gate. §Likelihood.

## A3 — Is 17/20 coverage grid mechanics or Davies pathology?
Mechanics: the fixed-A, fixed-ρ profile contains no unidentified parameter
(the Davies mechanism lives in the free-ρ detection statistic, not here);
the three misses land just below truth (0.6–0.9×), the signature of
crossing-resolution bias on a coarse grid; the replicate UL distribution is
otherwise well-behaved (median 1.2e-46). Remediation predeclared (B13):
replicate-calibrated critical value (Neyman-style) if the definitive study
confirms undercoverage; otherwise the limit is labeled non-coverage-
calibrated. §Upper limit.

## B1 — Does the coherence gate veto the true signal?
No: tripping the gate switches the band's *calibration*, never discards it.
A signal strong enough to raise measurable coherence is analyzed against the
matched null, where it stays anomalous — no admissible covariance converts
coherence *magnitudes* into mean-negative triple product. Cost is
sensitivity (wider matched-null tails), not exclusion. §Sign statistic.

## B2 — Specify the matched bootstrap.
Predeclared: diag + rank-1 null with |b_i b_j| reproducing the measured
|γ̂_ij| per bin and phases set to the t-*positive* configuration (maximal
admissible positive model-t at those coherences). A prescription, not a
least-favorable theorem: tail coverage at p<10⁻³ is validated by simulation
(importance sampling + GPD as production tools); bands whose matched null
cannot be certified carry no discovery weight. §Sign statistic.

## B3 — Foreground-inclusive null for the sign channel?
Adopted for production: at ET sensitivity the discovery p-value is calibrated
against diag + S_gw(f)Γ at the jointly fitted/astro-bounded foreground level
(which shifts the null negative and raises the bar). The universal diagonal
null remains correct for the MDC1 stretches analyzed here (no measurable
stochastic foreground; coherences at floor). §Sign statistic + rule.

## B4 — Two-channel discovery reach.
On the forecast figure now: ≈70 σ_F(T) (sign-limited coincidence), beside the
constrained-LR ≈30 σ_F(T) (conditional on witness-validated modeling) and the
1.645 σ_F UL curves. §Forecast + Fig. 6.

## B5 — Does the sign channel scale as T^{-1/2}?
Yes — demonstrated (run N2), not assumed: under the m^{3/2} weighting the
per-bin null fluctuation is m-independent while the cubic signal shift grows
as m^{3/2}(κA)³ → A_thr ∝ m^{-1/2} ∝ T^{-1/2}. Measured 50% multipliers:
69 σ_F at 64 s and 64 σ_F at 128 s (200 draws each; ratio 0.92 ≈ 1) —
constant in σ_F units across a factor 2 in T. This also refines the coarse
16-draw "84 σ_F" to ~70 σ_F everywhere it was quoted. §Forecast.

## B6 — Full-response strain injection.
Remains the decisive missing test and is stated as such (forecast labeling +
commissioning item (ii)); nothing further claimable without doing it.

## B7 — Amplitude calibration vs the UL proof.
Bound generalized: for G = diag(g_i), |g_i| ≥ 1−ε, GXG† ⪰ c(1−ε)²I whenever
X ⪰ cI, so Eq. (loewner) holds with A_h s_k(2−2ρ)(1−ε)² and the wall moves by
at most (1−ε)^{-2} ≈ 1+2ε — percent-level for percent-level calibration
priors, to be booked as a systematic when real budgets exist. §Upper limit.

## B8 — Is −1/2 exact for ET's 60° Michelsons?
Yes at DC: for coplanar detectors of any common opening angle rotated by σ,
tr(d₁d₂)/tr(d₁²) = cos 4σ; σ = 120° gives −1/2 exactly, independent of the
60° opening. Explicit-tensor numerical check in the validation suite. The
E1/E2/E3 convention mapping stays a commissioning item, but t is sign-flip
invariant, so the *sign* of t_gw is convention-free. §Sign statistic.

## B9 — Finite-frequency null stream.
Stated: the optimal null is the minimum-eigenvalue eigenvector of R Γ R†,
frequency-dependent once 2πfL/c and transfer asymmetries matter; building it
belongs to the full-response commissioning item; until then the equal-weight
null is a veto only, never a discovery ingredient. §Null stream.

## B10 — Line-forest stress envelope.
Run (N3): densities 0.3→3.0 lines/bin (up to 51/64 rank-≥2 bins), powers to
10×, and 10×-power combs with a forest on top: false rate 0.000 at 0.05 in
every case, and zero events at the 10⁻³ quantile in 2×10⁴ datasets. The
baseline density was a masked-forest caricature; the envelope makes the
choice immaterial. §Sign statistic.

## B11 — Predeclare the large-|r_k| removal rule.
Done: (i) leave-out-3 (the three largest-|r_k| bins, ≈5% of a 64-bin
analysis, calibration recomputed on survivors) must retain the discovery
p-value; (ii) band-split at the information median, each half at p<0.05.
§Sign statistic + rule.

## B12 — Witness misses weak broadband rank-2?
Defense in depth: whatever passes the witness-rank test must still pass the
*channel-based* coherence gate (blind to witness coverage), which caps
admitted contamination at the |γ̂|² ~ 1/m floor — where the measured
aggregate false rate at the discovery threshold is zero. The combined-screen
false-admission probability under a specified witness noise model is a
commissioning computation; the channel gate caps its consequences regardless.
§Rule.

## B13 — Plan if high-replicate coverage confirms undercoverage.
Predeclared: inflate the critical value by replicate calibration (choose Δ_c
for 95% empirical coverage — Neyman-style, affordable with the fast
pipeline); failing that, publish the limit explicitly labeled
non-coverage-calibrated. §Upper limit.

## B14 — Validate the production p<10⁻³ directly.
Run (N4): zero events at the 10⁻³ quantile in 3×10⁴ draws under
broadband-coherent rank-1 nulls (strengths 0.3–1.0, m to 10³) and 2×10⁴
line-forest datasets. Key structural finding: the third-round single-bin tail
concern does **not** propagate to the aggregated statistic — over 64 bins the
accumulated positive mean outruns the variance inflation — so the gate's real
work is protecting *localized* (few-bin) candidates, which the predeclared
leave-out/band-split requirements expose anyway. Foreground-inclusive-null
tails at 10⁻³ join the commissioning set. §Sign statistic.

## Code delivered with this round
- `qif_v2.py`: two-dimensional (amplitude, ρ) profile-scan seeding in
  `fit_model` (fixes the decaying-κ efficiency loss; null results unchanged).
- Measurement suite: `test-runs/rerun_results_v033.json` (N1–N5).
- Forecast figure: two-channel discovery curve added.
