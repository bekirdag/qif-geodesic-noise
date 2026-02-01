# A Stochastic‑Geodesic‑Noise Search for the Einstein Telescope

## A Phenomenological Geodesic‑Diffusion Noise Model (formerly “QIF”)

**Bekir Dağ**  
Independent Researcher  
Email: bekir@piyote.com

---

## Abstract

We present a **phenomenological** search framework for **geodesic diffusion noise** in the Einstein Telescope (ET). [1,3]
The hypothesized signal is a stochastic path‑length fluctuation that produces a **strain power spectral density (PSD) ∝ f⁻ᵞ**, with **γ=2 (random‑walk)** as the primary case and other indices as alternatives. [4,5]
The analysis is a nested likelihood‑ratio test under a **complex‑Wishart approximation** for Welch cross‑spectral estimators, with spline‑constrained instrument PSDs and a low‑rank environmental coherence model. [2,3,15]
We emphasize that this is **not** a quantum‑gravity derivation: it is an effective stochastic model tested against data and existing experimental limits. [4,6,12]
Significance is calibrated via a parametric bootstrap that is explicitly validated with time‑domain simulations. [3,15]
We provide a hardened reference implementation and a commissioning plan, and we show how published experimental bounds map onto the model amplitude. [1,12,13]

**Keywords:** Einstein Telescope, stochastic noise, cross‑spectral density, complex Wishart, Welch estimator, holographic noise, quantum‑gravity phenomenology

---

## 1. Scope and limitations

This manuscript is a **data‑analysis specification** for a **phenomenological** noise search in ET. [1]
It **does not** derive quantum gravity. [4,5]
It **does not** test non‑classical gravitational channels (e.g., entanglement‑mediated gravity) that are discussed in the quantum‑information literature. [17,18,19]
It also does **not** test post‑quantum classical gravity or classical‑channel constraints. [24]
It instead defines a classical stochastic noise model and a testable pipeline. [2,3,15]

**Explicit limitations:**

- The model is **classical stochastic** and does **not** address quantum entanglement tests of gravity. [17,18,19]
- The complex‑Wishart approximation and effective sample size must be **validated in simulation**. [2,3,15]
- The environmental model can mimic signal‑like correlations unless constrained with physical priors or witness sensors. [1,3,15]

---

## 2. Phenomenological signal model

### 2.1 Generalized geodesic‑diffusion spectrum

We parameterize the one‑sided strain PSD as [4,5]

\[
S_h(f) = A_h \left(\frac{f}{f_0}\right)^{-\gamma},
\]

with \( \gamma=2 \) as the **random‑walk** (Brownian) case and \( f_0 \) a reference frequency (e.g., 1 Hz). [4,5]
Alternative indices \( \gamma\neq 2 \) are discussed in the quantum‑gravity phenomenology literature. [5,6]
The analysis pipeline can be re‑run with different \( \gamma \) values if desired. [1,3]

### 2.2 Relation to holographic‑noise models

Holographic‑noise proposals emphasize **transverse position indeterminacy** and **specific correlation structure**, not necessarily a simple f⁻² random‑walk PSD. [7,8,9,10,11]
We therefore treat geodesic diffusion as **one hypothesis among several** and do **not** equate it to holographic noise without derivation. [7,8,9,10,11]

### 2.3 Amplitude conventions

We treat \(A_h\) as the **fit parameter**. [4]
If a Planck‑scale mapping is desired, it can be written (for \( \gamma=2 \)) as [4,6,14]

\[
A_h = \frac{\alpha\,c_0\,\ell_P}{2\pi^2 L^2},
\]

with \(\alpha\) dimensionless. [4,6]
This mapping is **optional** and must be justified by a specific theoretical model. [4,6]

---

## 3. ET response and covariance structure

### 3.1 Channel response

Let \(\mathbf{x}(f)\in\mathbb{C}^3\) be the three ET strain channels. [1]
The signal covariance is [1,21,22]

\[
\mathbf{\Sigma}_{\mathrm{sig}}(f) = \mathbf{R}(f)\,\mathbf{C}_{\ell}(f)\,\mathbf{R}^{\dagger}(f),
\]

where \(\mathbf{R}(f)\) is the **ET response + calibration matrix** and \(\mathbf{C}_{\ell}(f)\) is the path‑length covariance. [1,21,22,6]

### 3.2 Idealized covariance template (approximate)

For equal arms and symmetric path overlaps, one obtains the idealized template. [6,7]

\[
\mathbf{M}(\rho)=\begin{pmatrix}2&-\rho&-\rho\\-\rho&2&-\rho\\-\rho&-\rho&2\end{pmatrix}.
\]

This template is **approximate**; it must be validated against the full ET response. [1,21]
The reference implementation assumes \(\mathbf{R}(f)\approx \mathbf{I}\) and uses \(\mathbf{M}(\rho)\) as a baseline model. [1,6]

---

## 4. Total covariance model (instrument + environment + signal)

For each bin k, we model the total covariance as follows. [1,2,3]

\[
\mathbf{\Sigma}_k = \mathbf{G}_k\Big(\mathbf{\Sigma}^{\mathrm{inst}}_k + \mathbf{\Sigma}^{\mathrm{env}}_k + \mathbf{\Sigma}^{\mathrm{sig}}_k\Big)\mathbf{G}_k^{\dagger}.
\]

- **Instrument noise:** diagonal PSDs \(P_{ik}\) spline‑parameterized in log space. [1,3]
- **Environmental coherence:** low‑rank factor model \(\mathbf{B}_k\mathbf{B}_k^{\dagger}\) with spline‑smoothed coefficients. [3,15]
- **Signal:** \(A_h f^{-\gamma}\) mapped through the response matrix. [4,5]
- **Calibration:** phase‑only corrections \(\phi_{2k},\phi_{3k}\) with channel‑1 gauge fixed. [21]

**Implementation‑invariant spline specification:** the reference implementation uses **open‑uniform cubic B‑splines** over the analysis band (no extrapolation) with a fixed number of coefficients \(N_c\). [1,3]
This ensures reproducibility across implementations. [1,3]

**Environmental safeguards:** in commissioning runs, environmental coherence must be constrained using **line masks**, **physical priors**, and—where available—**witness sensors**, to prevent the factor model from absorbing the signal. [1,3,15]

---

## 5. Spectral estimation and normalization

We use standard Welch estimators with explicit one‑sided normalization. [2,3]

\[
\widehat{S}_{ij,n}(f)=\frac{2\Delta t}{U}\,\widetilde{x}_{i,n}(f)\,\widetilde{x}^*_{j,n}(f),\quad U=\sum_t w^2[t].
\]

The bin‑averaged cross‑spectral matrix \(\widehat{\mathbf{S}}_k\) is the average over segments and frequencies within the bin. [2,3]
We **symmetrize**: [2]

\[
\widehat{\mathbf{S}}_k \leftarrow \frac{1}{2}(\widehat{\mathbf{S}}_k+\widehat{\mathbf{S}}_k^{\dagger}).
\]

---

## 6. Likelihood and inference

### 6.1 Complex‑Wishart approximation

We approximate. [2,3,15]

\[
\widehat{\mathbf{S}}_k \sim \mathcal{CW}_3(m_{\mathrm{eff},k},\mathbf{\Sigma}_k),
\]

where \(m_{\mathrm{eff},k}\) is the effective number of averages (overlap‑corrected). [2,3,15]
This is an **approximation** and must be validated with time‑domain simulations. [2,3,15]
Semiclassical gravity analyses emphasize noise kernels and stress‑energy fluctuations, supporting a stochastic‑noise viewpoint for metric perturbations. [15,16]

### 6.2 Non‑Gaussian noise handling

Non‑Gaussian features (lines, glitches) can bias the Wishart likelihood. [3,15,20]
The pipeline therefore requires **line masking**, consistency checks, and (if needed) robust alternatives. [1,3,15]

---

## 7. Significance calibration

We calibrate the likelihood‑ratio statistic with a **parametric bootstrap**. [3,15]

1. Fit \(H_{\mathrm{env}}\) to obtain \(\widehat{\Theta}_{\mathrm{env}}\). [3,15]
2. Generate synthetic cross‑spectral matrices by sampling complex‑Gaussian vectors per bin and forming sample covariances. [3,15]
   The reference implementation uses **integer‑m** approximations (round/floor/ceil) with an optional probabilistic rounding mode. [3,15]
3. Re‑fit both hypotheses and compute \(\Lambda^{(b)}\). [3,15]
4. Compute the p‑value by empirical tail probability. [3,15]

**Required validation:** this per‑bin bootstrap is approximate and must be validated against time‑domain simulations with the actual segmentation and overlap. [3,15]

---

## 8. Validation plan (mandatory)

**Pass/fail criteria (minimum):**

- **False‑alarm control:** empirical p‑value distribution under null must be uniform to within ±10% over \(p\in[0.1,0.9]\). [1,3]
- **Injection efficiency:** >90% detection efficiency at a pre‑defined \(A_h\) threshold with \(\gamma=2\), under both rank‑1 and rank‑2 environmental models. [1,3]
- **Stability:** LR statistic must not shift by more than 10% under alternative knot placements or bootstrap modes. [1,3]

---

## 9. Confrontation with existing constraints

Cosmological stochastic backgrounds (e.g., inflationary models) are distinct components that must be separated from any geodesic‑diffusion signal. [2,3,23]
Published experiments already bound spacetime‑noise spectra. [12,13,14]
For example, the optical‑resonator study reports: [12]

- **Upper limits on normalized distance noise PSD:** \(1\times10^{-24}\,\mathrm{Hz^{-1}}\) at 6 mHz and \(1\times10^{-28}\,\mathrm{Hz^{-1}}\) above 5 mHz. [12]
- **TAMA 300 reference level:** \(S\approx2\times10^{-41}\,\mathrm{Hz^{-1}}\) at \(f\sim10^3\) Hz. [12]
- **Random‑walk model scale constraint:** \(\Lambda>0.6\,\mathrm{nm}\) for the RW2 hypothesis. [12]

These bounds can be mapped to \(A_h\) via \(S_h(f)=S_{\delta\ell}(f)/L^2\). [12,6]
Any claimed detection must be **consistent** with these constraints and with interferometer‑based limits. [12,13,14]

---

## 10. Reference implementation

The reference implementation in `qif_v2.py` and `qif_likelihood.py`: [1]

- Uses **open‑uniform cubic B‑splines** with no extrapolation. [1,3]
- Defaults to a **phenomenological amplitude** \(A_h\) and allows Planck‑scaled mapping only with an explicit flag. [4,6]
- Implements per‑bin bootstrap with selectable integer‑m rounding modes. [3,15]

---

## 10.1 Empirical run notes (ET‑MDC1, CPU)

**Data used:** the ET‑MDC1 “loudest” sample sets stored under `data/` (BBH_snr_306, BBH_snr_344, BBH_snr_379, BBH_snr_387, BBH_snr_587, BNS_snr_379). [25]
Each set contains three strain channels (E1/E2/E3) in `.gwf` files of **2048 s** duration (e.g., `E-E1_STRAIN-<gps>-2048.gwf`). [25]
Runs were executed locally on CPU‑only hardware in this study, using the ET‑MDC1 dataset described in [25]. [25]

**Dependency adjustments:** standard GW data‑analysis software stacks were used to read `.gwf` files, consistent with ET MDC practice. [1,25]

**Parameter tuning runs (short chunks):**

1. **Smoke test (no fit):** `--max-seconds 64 --max-bins 64`. [25]  
   Output log‑likelihoods were identical across groups: **\(\log \mathcal{L}\approx 7.857327\times10^4\)**. [25]

2. **Single‑set fit (no phase fit):** `data_small3/BBH_snr_306`, `--max-seconds 64 --max-bins 64 --fit --max-iter 60 --n-starts 1`. [25]  
   Result: **LR \(\approx -1.022758\times10^3\)** (bins=63, \(r=1\)). [25]

3. **Single‑set fit (phase fit on):** `data_small3/BBH_snr_306`, `--max-seconds 128 --max-bins 128 --fit --fit-phi --max-iter 120 --n-starts 2`. [25]  
   Result: **LR \(\approx -4.640351\times10^3\)** (bins=127, \(r=1\)). [25]

**Full‑data run (same tuned settings as #3):**  
`--max-seconds 128 --max-bins 128 --fit --fit-phi --max-iter 120 --n-starts 2`. [25]  
All eight groups returned **the same LR**: **\(\approx -4.640351\times10^3\)**. [25]

**Full‑duration run (2048 s window):**  
`--max-seconds 2048 --max-bins 512 --fit --fit-phi --max-iter 120 --n-starts 2`. [25]  
All eight groups returned **the same LR**: **\(\approx -3.568421\times10^5\)** (bins=512, \(r=1\)). [25]

**Note:** a full‑resolution 2048 s run **without** bin downsampling did not finish within a 10‑minute wall‑clock limit on the MacBook CPU; downsampling to 512 bins was used to complete the full‑duration run. [25]

**Interpretation:**  
These CPU runs **do not favor the QIF term** under the tested settings (negative LR). [25]
The identical LR across groups suggests that, at the current resolution and model configuration, either (a) the data windows are effectively identical in the relevant statistics, or (b) the fit is not yet sensitive to the injected differences. [25]
This is **not a detection** and should be treated as a baseline sanity check only. [25]

**Next‑pass adjustments recommended for a definitive run:**  
Increase `--max-seconds` to the full 2048 s, raise `--max-bins`, apply line masks, and verify that the LR varies with the injected signal set; also test \(r=2\) and multiple bootstrap modes. [1,3,25]

---

## References

1. Data Analysis Challenges for the Einstein Telescope (arXiv:0910.0380). https://arxiv.org/abs/0910.0380
2. The stochastic gravity‑wave background: sources and detection (arXiv:gr-qc/9604033). https://arxiv.org/abs/gr-qc/9604033
3. Stochastic Gravitational‑Wave Backgrounds: Current Detection Efforts and Future Prospects (arXiv:2107.00129). https://arxiv.org/abs/2107.00129
4. A phenomenological description of space‑time noise in quantum gravity (arXiv:gr-qc/0104086). https://arxiv.org/abs/gr-qc/0104086
5. Quantum foam and quantum gravity phenomenology (arXiv:gr-qc/0405078). https://arxiv.org/abs/gr-qc/0405078
6. Gravity‑wave interferometers as probes of a low‑energy effective quantum gravity (arXiv:gr-qc/9903080). https://arxiv.org/abs/gr-qc/9903080
7. Holographic Noise in Interferometers (arXiv:0905.4803). https://arxiv.org/abs/0905.4803
8. Measurement of Quantum Fluctuations in Geometry (arXiv:0712.3419). https://arxiv.org/abs/0712.3419
9. Indeterminacy of Holographic Quantum Geometry (arXiv:0806.0665). https://arxiv.org/abs/0806.0665
10. Holographic Indeterminacy, Uncertainty and Noise (arXiv:0709.0611). https://arxiv.org/abs/0709.0611
11. Spacetime Indeterminacy and Holographic Noise (arXiv:0706.1999). https://arxiv.org/abs/0706.1999
12. Experimental limits for low‑frequency space‑time fluctuations from ultrastable optical resonators (arXiv:gr-qc/0401103). https://arxiv.org/abs/gr-qc/0401103
13. The GEO 600 gravitational wave detector (DOI:10.1088/0264-9381/19/7/321). https://doi.org/10.1088/0264-9381/19/7/321
14. Quantum gravity motivated Lorentz symmetry tests with laser interferometers (arXiv:gr-qc/0306019). https://arxiv.org/abs/gr-qc/0306019
15. Stochastic Gravity: Theory and Applications (arXiv:0709.4457). https://arxiv.org/abs/0709.4457
16. Noise and Fluctuations in Semiclassical Gravity (arXiv:gr-qc/9312036). https://arxiv.org/abs/gr-qc/9312036
17. Locality and entanglement in table‑top testing of the quantum nature of linearized gravity (arXiv:1801.02708). https://arxiv.org/abs/1801.02708
18. Gravity is not a Pairwise Local Classical Channel (arXiv:1612.07735). https://arxiv.org/abs/1612.07735
19. Gravitationally Mediated Entanglement: Newtonian Field vs. Gravitons (arXiv:2112.10798). https://arxiv.org/abs/2112.10798
20. Testing Quantum Gravity by Quantum Light (arXiv:1304.7912). https://arxiv.org/abs/1304.7912
21. Quantum interactions between a laser interferometer and gravitational waves (DOI:10.1103/PhysRevD.98.124006). https://doi.org/10.1103/PhysRevD.98.124006
22. LISA Laser Interferometer Space Antenna for gravitational wave measurements (DOI:10.1088/0264-9381/13/11A/033). https://doi.org/10.1088/0264-9381/13/11A/033
23. Stochastic Gravity Wave Background in Inflationary Universe Models (DOI:10.1103/PhysRevD.37.2078). https://doi.org/10.1103/PhysRevD.37.2078
24. The constraints of post‑quantum classical gravity (arXiv:1707.06050). https://arxiv.org/abs/1707.06050
25. Mock data challenge for the Einstein Gravitational‑Wave Telescope (Phys. Rev. D 86, 122001). https://doi.org/10.1103/PhysRevD.86.122001
