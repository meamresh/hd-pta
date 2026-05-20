# Methods

A walkthrough of the physics and statistics behind `hd-pta`: how a
pulsar timing array detects a gravitational-wave background, and what
each stage of the pipeline actually computes.

---

## 1. The idea

A pulsar is a rotating neutron star whose radio beam sweeps past Earth
with extraordinary regularity. The arrival time of each pulse can be
predicted by a **timing model** — spin frequency and spin-down,
astrometric position and proper motion, binary-orbit parameters,
dispersion in the interstellar medium. Subtracting the model prediction
from the measured arrival times leaves the **timing residuals**: what
the model could not account for.

A gravitational wave passing between Earth and a pulsar perturbs the
arrival times. A *stochastic background* of such waves — the incoherent
sum of many sources, most plausibly inspiralling supermassive black hole
binaries throughout the universe — leaves a faint red-spectrum signal in
every pulsar's residuals.

The problem: that signal is far smaller than each pulsar's own intrinsic
red noise. No single pulsar reveals it. The detection lives entirely in
the **correlations between pulsars**.

---

## 2. The Hellings–Downs curve

Hellings & Downs (1983) showed that an isotropic, unpolarized GW
background induces a correlation between any two pulsars that depends
**only on the angle between them**, `ζ`:

```
Γ(ζ) = (3/2) x ln(x) − x/4 + 1/2,      x = (1 − cos ζ) / 2
```

with the convention `Γ(0⁺) = 1/2`. The shape is quadrupolar: pulsars
close together on the sky are positively correlated, pairs near 90°
are anti-correlated, pairs near 180° are positively correlated again.

This shape is the fingerprint. Competing systematics produce *different*
angular patterns:

- a **clock error** common to all pulsars → monopole (flat, `ℓ = 0`)
- a **solar-system ephemeris error** → dipole (`ℓ = 1`)
- a **GW background** → the specific quadrupole-dominated Hellings–Downs
  curve

Recovering the Hellings–Downs shape — rather than a monopole or dipole —
is what distinguishes a gravitational-wave detection from an artifact.

---

## 3. Removing the timing model

The raw residuals still carry covariance with the timing-model
parameters: any GW or noise analysis must account for the fact that the
timing model was *fitted* to the same data.

Linearized about the best-fit solution, the timing model is a design
matrix `M` (one column per timing parameter). The component of the
residuals lying in the column space of `M` has been absorbed by the fit.
The pipeline removes it by projecting onto the orthogonal complement:

```
r_clean = [ I − M (Mᵀ W M)⁻¹ Mᵀ W ] r
```

where `W` is the diagonal matrix of inverse-squared TOA uncertainties.
This is the standard analytic marginalization over timing-model
parameters; `enterprise` performs the equivalent operation internally
(`tm_svd=True` uses a numerically stable SVD form).

---

## 4. Frequency-domain reduction

Pulsars are observed on different, irregular cadences over different
baselines, so residuals cannot be cross-correlated sample by sample.
Instead each pulsar's cleaned residuals are projected onto a **common
Fourier basis** at frequencies

```
f_k = k / T_span,      k = 1, 2, …, 14
```

where `T_span` is the total observing baseline of the array (~16 yr for
NANOGrav 15-yr). Fourteen modes capture the low-frequency band where the
GW background dominates. The projection is an inverse-variance-weighted
least-squares fit, so noisier epochs contribute less.

The GW background has a red (steeply falling) spectrum. For a population
of circular, GW-driven supermassive black hole binaries the
characteristic strain follows `h_c ∝ f^(-2/3)`, which corresponds to a
timing-residual power spectral density

```
P(f) = A² / (12 π²) · (f / f_yr)^(−γ) · f_yr^(−3),     γ = 13/3
```

`A` is the dimensionless amplitude at a reference frequency of
`f_yr = 1/yr`. NANOGrav 15-yr measured `A ≈ 2.4 × 10⁻¹⁵`.

---

## 5. The optimal statistic

For each pair of pulsars `(a, b)` the **optimal statistic** forms a
cross-correlation estimator `ρ_ab` together with its uncertainty
`σ_ab`. Under a GW background,

```
⟨ρ_ab⟩ = A² · Γ(ζ_ab)
```

i.e. each pair's correlation is the GW amplitude squared times the
Hellings–Downs value for that pair's angular separation. Two products
follow:

- **Amplitude.** A weighted fit of `ρ_ab` against `Γ(ζ_ab)` across all
  pairs estimates `A²`.
- **Significance.** The signal-to-noise ratio of that fit is the
  Hellings–Downs detection significance. NANOGrav 15-yr reports ~3.5–4σ
  from the full 67-pulsar array (2,211 pairs).

Plotting `ρ_ab / A²` against angular separation places every pair on the
dimensionless Hellings–Downs curve — this is the figure the animation
builds, pair by pair.

Signal-to-noise grows roughly with the number of pulsar **pairs**, which
scales as the square of the number of pulsars. This is why a subset of
the array (used here when RAM is limited) yields a genuinely
lower-significance, noisier curve: 22 pulsars give 231 pairs, the full
67 give 2,211.

---

## 6. Noise marginalization

Each pulsar has its own intrinsic red noise, with amplitude and spectral
index that are **not known exactly**. The NANOGrav analysis produced a
Bayesian posterior — an MCMC chain — over all pulsars' red-noise
parameters. This repository ships a presampled chain
(`curn_14f_pl_vg.core`).

The optimal statistic is evaluated once per posterior sample and the
results are averaged. The final `ρ_ab`, `A²`, and significance are
therefore **marginalized over** the noise uncertainty rather than
conditioned on a single guess.

This step is not optional. Fixing the red-noise parameters at a single
point estimate can bias the correlations and inflate the apparent
Hellings–Downs signal. The `--n-noise` argument sets how many posterior
samples are drawn:

- the per-pair `ρ_ab` values are fairly stable even for small `N`
  (they are dominated by the data, not the noise draw)
- the significance histogram tightens as `N` grows
- NANOGrav's published analysis uses `N = 10000`

This marginalization is the reason the real-data pipeline requires the
full `enterprise` machinery, while the simulation — where the injected
noise model *is* known exactly — does not.

---

## 7. Simulation vs. real data

`simulate_hd.py` and `ng15_optimal_statistic.py` run the same conceptual
analysis on different inputs.

| | Simulation | Real data |
|---|---|---|
| Pulsar positions | synthetic, galactic-plane-biased | real NANOGrav sky |
| Residuals | injected GWB + injected noise | real NANOGrav 15-yr TOAs |
| Noise model | known exactly | marginalized over MCMC posterior |
| Detection significance | high (idealized upper bound) | ~3.5–4σ (full array) |
| Purpose | see the curve clearly | the genuine measurement |

The simulation's significance is higher than the real result *by
construction*: knowing the noise model exactly removes the dominant
source of real-world uncertainty. It is the best case, useful for
visualization and for validating the pipeline against a known injected
amplitude. The real-data run is the honest measurement.

---

## 8. An implementation pitfall worth recording

Generating a time-domain noise realization from a one-sided power
spectral density `P(f)` (used in the simulation) requires a specific FFT
normalization. For an `N`-sample series with spacing `dt`, the real and
imaginary parts of each `rfft` coefficient must be drawn with

```
σ = sqrt( N · P(f_k) / (4 · dt) )
```

The factor of `N` is easy to drop. Doing so does not crash anything — it
silently rescales every noise component by a constant, so the GWB,
intrinsic red noise, and white noise all come out at the wrong absolute
level while still *looking* plausible. The simulation guards against
this with an explicit check that the realized per-pulsar residual RMS
matches the value obtained by integrating `P(f)` over the analyzed band.

The general lesson: in any PSD-to-time-domain step, verify the realized
variance against the analytic integral of the spectrum before trusting
anything downstream.

---

## References

- R. W. Hellings & G. S. Downs (1983), *Upper limits on the isotropic
  gravitational radiation background from pulsar timing analysis*,
  ApJL 265, L39.
- Agazie et al. (NANOGrav, 2023), *The NANOGrav 15-year Data Set:
  Evidence for a Gravitational-Wave Background*, ApJL 951, L8.
- Agazie et al. (NANOGrav, 2023), *The NANOGrav 15-year Data Set:
  Observations and Timing of 68 Millisecond Pulsars*, ApJL 951, L9.
- `enterprise` / `enterprise_extensions`,
  github.com/nanograv/enterprise_extensions.
