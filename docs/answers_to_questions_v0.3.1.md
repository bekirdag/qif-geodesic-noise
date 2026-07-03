# Answers to the third-round (v0.3.1) referee questions → v0.3.2

All items resolved in `docs/et_geodesic_noise_paper_v0.3.pdf` (v0.3.2, 2026-07-03).
Definitive third-round numbers: `test-runs/rerun_results_v032.json`. Referee A =
the four "sticky questions"; Referee B = the fourteen numbered questions.

## A1 — Is the residual deviance riding on the ≤6.9-unit optimization noise floor?
**No — measured directly.** On fixed data with independent multistart seeds, the
nested statistic reproduces to Λ = 29.7 ± 0.2 (spread 0.58 over 8 seeds) at
32 σ_F: the residual signal deviance is ~150× the optimization noise on Λ. The
6.9-unit figure is the scatter of a *single fit's absolute* lnL; Λ is a
difference of cross-pollinated fits and its errors cancel. The null side is
dominated by the (calibrated) Davies tail, not convergence noise: one fixed
null draw ranged over {≈0 (7 seeds), 15.6 (1 seed)} — the same tail the n=100
bootstrap maps to Λ_95 = 4.53. Paper §Injections.

## A2 — Politics of "dead on arrival".
Honesty kept, framing added (§Bounds): no tabletop can measure the L-scaling
exponent η by itself; ET is the **irreplaceable long-baseline anchor** of the
(tabletop, ET) η measurement, and this framework is the anchor's analysis
machinery.

## A3 — Does a joint CBC-foreground fit collapse the remaining information?
**No — measured at the Fisher level (new run R2).** Adding a foreground
amplitude nuisance (f^{-7/3}, triangle correlation −1/2 fixed by geometry) to
the full 220-parameter profiled Fisher: σ_F^prof goes 2.01e-48 → 4.93e-48,
a further **×2.45 — not a collapse**. The surviving handles (spectral index
×4.6 across the band, null stream, rank) carry real weight. Free-index
foreground and realistic simulations remain commissioning items.

## A4 / B11 — Localized rank-2 glitches and realistic line forests vs T_sign.
Multi-line MC (Poisson forest, mean 2.5 rank-≥2 bins per dataset): aggregate
false rate **0.000** at nominal 0.05 — the rank-1 majority's positive-mean push
dominates. Against a *localized* rank-2/3 facility artifact the defense is
locality, still fit-free: per-bin contributions m^{3/2}r_k are inspectable, and
a candidate must survive band-split consistency, largest-|r_k|-bin removal, and
the new coherence gate (which a localized rank-2 event trips in its band).
Paper §Sign statistic (line/glitch bullet) + detection rule.

## B1 / B14 — Are the sensitivity curves valid under the strain reading?
**Conceded and relabeled — the referee's strongest question.** Every curve is
now explicitly a **path-template forecast (R = I)**: exact under the
path-length reading, provisional under the strain reading pending
full-response injections (R C R† injected, path-template recovered, bias
quantified — commissioning item (ii)). §Forecast opening paragraph + figure
caption + abstract.

## B2 — S_h terminology.
Fixed at Eq. (1): S_h is the strain-equivalent PSD of a **single optical
path**; the measured per-channel signal PSD is 2S_h; "every amplitude in this
paper is per-path unless stated otherwise."

## B3 — Conservativeness for ALL diag+rank-1 at finite m.
**Proved for the mean, refuted for the tail, gated in production (run R3).**
Exact Isserlis expansion: E[t̂] = c_m·t_Σ + (1/m)(1−1/m)[|Σ12|²Σ33 + |Σ13|²Σ22
+ Σ11|Σ23|²] + (1/m²)[… ≥ Σ11Σ22Σ33], every term non-negative on the class →
mean-conservative. But a 200-config scan finds worst-case single-bin tail
false rate **0.24** at nominal 0.05 (coherent factor, m=10³): coherence
inflates the *variance* of the ratio statistic. Production prescription: the
universal p-value is quoted only behind a **fit-free coherence gate** (all
|γ̂_ij|² at the 1/m floor in contributing bins); MDC1 passes everywhere, so
the published p-values stand. The prior blanket "conservative under the entire
class" claim is corrected in the revision record.

## B4 — Fractional m_eff in the calibration.
Fixed in code: `sign_channel_pvalue` passes fractional m_eff directly — the
Bartlett construction takes non-integer Gamma shapes (m, m−1, m−2), matching
the fractional-Wishart moments the m_eff prescription implies. Rounding bias
was O(1/2m) per bin (negligible at m ≳ 70 but wrong in principle).
Time-domain closure of the Welch→Wishart map remains the top commissioning
item.

## B5 — Figure 4 vs Table 2 threshold inconsistency.
**Real; fixed.** The figure's LR detection curve is regenerated with the
bootstrap-calibrated Λ_95 = 4.53 (same as Table 2) and the legend states it.

## B6 — "Available matches Fisher" overstated.
Qualified: matches at small amplitude (12.5 vs 16 at 4σ_F), grows
sub-quadratically at large amplitude where the local expansion must fail
(log-determinant saturation).

## B7 — f_min sensitivity.
New sweep (run R6): σ_F changes ×1.8 end-to-end over f_min ∈ {5,10,15,20,30} Hz
(1.48–2.69e-48) and the Asimov ratio at fixed 64σ_F stays Λ = 49–78 — the
threshold in σ_F units is cutoff-robust. Median-information frequency: 22 Hz.

## B8 — Delay corrections above 1 kHz.
Fraction of Fisher information above 1 kHz: **3.7e-6** (2.2e-4 above 500 Hz).
Delayed-correlation corrections are irrelevant for γ = 2. §Fisher.

## B9 — Floating-ρ reporting.
Policy fixed in §β: headline quantity is β = ρA_h; limits published as a
fixed-ρ family {0.2, 0.5, 0.8} (A_h(ρ) ≈ β^95%/ρ); detection significance via
the ρ-profiled Λ with bootstrap calibration (which pays the Davies price).

## B10 — Numerical environmental-rank criterion.
Defined in the rule: per band, the second generalized eigenvalue of the
witness-projected channel cross-spectral matrix must be consistent at 95% with
its time-slide null; failing bands lose sign-channel discovery weight but stay
in the UL analysis (whose proof never references rank).

## B12 — Coverage of the new UL.
First empirical check (run R9): 20 Wishart replicates at A_true = 7.8e-47
through the full UL procedure → **17/20 cover** (0.85; binomial 95% CI ≈
[0.62, 0.97]). Mild undercoverage not excluded; coarse grid biases low; the
definitive study (finer grid, O(10²) replicates, non-Gaussian noise) is a
commissioning gate and the limit carries that caveat.

## B13 — Two-channel effective threshold.
Stated in the rule: the coincidence requirement makes the **sign channel the
practical discovery threshold (~84 σ_F at 64 s)**; an LR-only candidate below
it is a witness-model-conditional measurement, not a detection.

## New caveat surfaced by us, not the referees (run R10).
An isotropic SGWB in the ET triangle has pairwise correlation coefficient
**−1/2** in this covariance normalization (tr(d₁d₂)/tr(d₁²) = cos 4σ at
σ = 120°), so it *also* produces Re t < 0: the sign channel establishes
**non-instrumental correlated power**, not geodesic origin. Geodesic/GW
separation belongs to the null stream (∝(1−ρ)) and the spectral index — as the
detection rule already requires. Stated in §Sign statistic and the abstract.

## Code delivered with this round
- `qif_v2.py`: fractional-m sign-channel calibration (no rounding).
- Measurement suite results: `test-runs/rerun_results_v032.json` (R1–R10).
- Figure 4 regenerated with the calibrated LR threshold.
