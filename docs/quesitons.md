## Merged, vetted questions to ask the author

### 1. Signal covariance derivation

Can you derive the matrix (M(\rho)), including the signs, diagonal factor of 2, off-diagonal structure, and the prefactor (1/2), from an explicit arm/path-length diffusion model? If (M(\rho)) is purely phenomenological, which physically plausible geodesic-diffusion covariance structures are being intentionally excluded?

### 2. Physical meaning of (\rho(f))

Is (\rho(f)) meant to be a physical ET geometry/beam-sharing response, a nuisance flexibility parameter, or a phenomenological proxy for an unknown correlation law? If it is physical, why not impose the derived ET response; if it is free, how do you control the extra look-elsewhere freedom introduced by a spline-parameterized signal correlation?

### 3. (A_h)–(\rho) degeneracy

Since the diagonal part of the signal template is largely degenerate with the instrument PSDs and the identifiable part is mainly the off-diagonal component proportional to (\rho A_h), can you show the joint profile likelihood in ((A_h,\rho))? What prevents a ridge where changes in (\rho) are compensated by changes in (A_h)?

### 4. The (\rho\to1) null-stream limit

Your null-stream discriminator loses geodesic-diffusion power as (\rho\to1). Is there a physical prior or geometric argument that prevents (\rho) from approaching this limit? If not, how would the search distinguish a highly correlated geodesic-diffusion component from a signal class that cancels almost completely in the null stream?

### 5. Full ET response bias

Can you quantify the bias in (\hat A_h), (\hat\rho), and (\Lambda) caused by using (R(f)\approx I) instead of a frequency-dependent ET response? A useful test would be: inject with a realistic (R(f)C_\ell(f)R^\dagger(f)), recover with the current (M(\rho)), and report the bias.

### 6. Gauge/sign convention in the off-diagonal argument

The identifiability argument relies on the signal having three negative real off-diagonal terms that a rank-1 environmental factor cannot reproduce. How is this statement invariant under channel sign conventions and phase calibration choices? With per-channel phase rotations, can an environmental factor be rotated into a signal-like sign pattern?

### 7. Rank-2 environmental stress test

The paper argues identifiability against a rank-1 environmental coherence model, but mentions (r=2) only as a stress test. What are the actual (r=2) results? Does the detection threshold degrade modestly, or can the rank-2 model absorb most of the geodesic-diffusion template?

### 8. Calibration nuisance dimensionality

The notation (\phi_{2k},\phi_{3k}) suggests per-frequency-bin phase calibration parameters. Are these independent per bin, or smooth constrained curves? If they are per-bin free parameters, can you show an ablation comparing per-bin phases against low-order spline or physically motivated calibration models?

### 9. Missing amplitude calibration uncertainty

Why are calibration nuisances phase-only? Real calibration errors also include amplitude response errors. Are amplitude errors fully absorbed by the diagonal PSD splines, or can they bias the cross-spectral structure used to estimate (A_h)?

### 10. Spline basis versus broadband signal absorption

You showed that too few PSD spline coefficients can create a fake signal. What prevents too many coefficients, or different knot placements, from subtracting part of a genuine (f^{-2}) broadband signal? Please show (\Lambda), (\hat A_h), and injection recovery as functions of (N_c) and knot placement.

### 11. High-Q line leakage

How does the pipeline behave in the presence of dense, high-Q instrumental lines and their spectral leakage wings? A smooth (N_c=20) PSD spline cannot model line wings directly, so do line masks plus Hann-window leakage suppression reduce the projection of unresolved line power onto the (f^{-2}) template below the target sensitivity?

### 12. Frequency-bin correlations

The likelihood treats binned cross-spectral matrices as independent complex-Wishart draws. With Hann windows, overlap, bin averaging, and spectral leakage, neighboring frequency bins are not strictly independent. Have you measured how much this frequency-bin covariance changes the LR null distribution and the bootstrap tail?

### 13. Welch-overlap Wishart approximation

The scalar (m_{\rm eff,k}) correction adjusts the variance of overlapped Welch averages, but it may not reproduce the full complex-Wishart likelihood shape. Can you validate the LR distribution using time-domain Gaussian simulations passed through the exact same Hann/50%-overlap Welch pipeline?

### 14. Bootstrap validity near a boundary

At (A_h=0), parameters such as (\rho) are weakly identified or unidentified. This is more complicated than the simple (\frac12\chi^2_0+\frac12\chi^2_1) boundary case. Have you checked bootstrap p-value convergence as a function of (n_b), especially in the tail relevant for detection claims?

### 15. Parametric bootstrap under model misspecification

The current parametric bootstrap draws from the fitted Wishart null, so it reproduces optimizer variance but not glitches, line nonstationarity, CBC residuals, or slow PSD drift. Can you compare it with time-domain simulation, block bootstrap, or real clean-noise null stretches to measure false-alarm calibration under model misspecification?

### 16. Time-domain injection recovery

The injection study uses Wishart resamples conditioned on a fitted environmental model. That validates part of the likelihood machinery, but not frame-level effects such as windowing, line leakage, nonstationarity, or CBC subtraction residuals. Can you run time-domain injections through the full pipeline and provide detection-efficiency curves?

### 17. BNS outlier null-stream decomposition

For the BNS_snr_379 outlier with (\Lambda=51.52), what does the null-stream decomposition show numerically? If the excess is really CBC contamination, subtracting or projecting the CBC contribution should reduce (\Lambda) toward the null distribution.

### 18. Clean-noise or CBC-subtracted validation

Can you provide at least one validation table on signal-free ET-MDC1 stretches or CBC-subtracted stretches? The current “loudest” segments are useful for shakedown, but they cannot establish the false-alarm behavior of a production stochastic search.

### 19. Robustness of the null stream

The simple null stream (E_1+E_2+E_3) assumes ideal equal response and calibration. With realistic ET response, calibration errors, and unequal transfer functions, what is the optimal null combination, and what leakage floor remains for ordinary gravitational-wave signals?

### 20. Coverage of the profile-likelihood upper limit

Does the quoted 95% profile-likelihood upper limit have correct empirical coverage under non-Gaussian or intermittent noise? Please test whether glitches and slow PSD variations make the limit over-cover, under-cover, or become biased high.

### 21. Low-frequency cutoff dependence

For (\gamma=2), most sensitivity comes from the low-frequency part of the band, where seismic, Newtonian, suspension, and environmental couplings are hardest. How do (\hat A_h), (\Lambda), and the forecasted limit change when the lower cutoff is moved from 10 Hz to, for example, 5, 15, 20, or 30 Hz?

### 22. Long-term (T_{\rm obs}^{-1/2}) forecast

How will one year of data actually be combined? Will (A_h) be common across all segments while PSD and environmental nuisances are segment-dependent? At what level of slow instrumental drift does the (T_{\rm obs}^{-1/2}) scaling break and become a systematic floor?

### 23. Astrophysical foreground confusion

How would an unresolved astrophysical stochastic background, anisotropic foreground, or imperfectly subtracted CBC population bias the geodesic-diffusion search? With realistic ET response, do these backgrounds remain orthogonal to the null-stream/geodesic template, or is a joint foreground model required?

### 24. Existing-bound comparison

Can you provide a table mapping each external bound into (A_h), explicitly stating one-sided versus two-sided PSD convention, strain versus displacement interpretation, reference frequency (f_0), and arm-length assumptions? This is especially important because the Planck-scale comparison depends strongly on whether the noise is treated as strain noise or path-length diffusion.

### 25. Pre-declared detection rule and trials factor

If future analyses scan over (\gamma), (\rho(f)), spline richness, knot placements, segment choices, line masks, or environmental rank, how will the trials factor be handled? What exact pre-declared combination of LR, bootstrap p-value, null-stream behavior, injection recovery, and environmental vetoes would constitute a candidate detection?

### 26. Computational scaling

A production analysis with full-refit bootstrap, rank-2 environmental factors, smooth calibration models, many frequency bins, and long-duration data could be expensive. What is the projected computational cost for (10^3)–(10^4) bootstrap replicates, and what approximations would preserve valid p-values if full refits become prohibitive?

### 27. Optimizer diagnostics

Given the serious v0.1 optimizer failure, can you report optimizer diagnostics for every headline result: gradient/KKT residuals, number of boundary hits, multi-start stability, cross-backend agreement, and whether re-seeding from the alternative ever improves the null nuisance optimum?

These are the questions I would actually hand to the author. They are hard, but each one targets a real open vulnerability in the current v0.2 framework rather than restating a limitation the paper already admits.

