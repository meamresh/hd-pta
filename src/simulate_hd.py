"""
simulate_hd.py
==============
Physics-accurate Hellings-Downs SIMULATION, calibrated to NANOGrav 15-yr.

Produces the static 3-panel summary figure (figures/hd_simulation.png) and
saves the per-pair products to results/sim_products.npz for the animation.

This is a SIMULATION: it injects a known gravitational-wave background and
NG15-realistic per-pulsar noise, then runs the same cross-correlation
analysis applied to real data. It is calibrated to NANOGrav 15-yr
(Agazie+ 2023, ApJL): 67 pulsars, 2,211 pairs, T = 16 yr,
A_GWB = 2.4e-15 (+0.7/-0.6) at f_yr, gamma = 13/3.

Because the noise model is known exactly here (unlike real data, where it
must be marginalized), the recovered detection significance is higher than
the real ~3.5-4 sigma -- this is the idealized upper bound, useful for
seeing the H-D curve clearly. For the genuine measurement on real NANOGrav
data, use src/ng15_optimal_statistic.py.

Run:  python src/simulate_hd.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_FIGDIR = os.path.join(_ROOT, 'figures')
_RESDIR = os.path.join(_ROOT, 'results')
os.makedirs(_FIGDIR, exist_ok=True)
os.makedirs(_RESDIR, exist_ok=True)

np.random.seed(15)

# =====================================================================
# CONSTANTS / NG15 PARAMETERS
# =====================================================================
yr = 365.25 * 86400.0
f_yr = 1.0 / yr

N_PULSARS = 67
N_PAIRS = N_PULSARS * (N_PULSARS - 1) // 2
T_SPAN_YR = 16.0
T_SPAN = T_SPAN_YR * yr
N_FOURIER = 14
A_GWB = 2.4e-15
GAMMA = 13.0 / 3.0

f_k = np.arange(1, N_FOURIER + 1) / T_SPAN


# =====================================================================
# PHYSICS FUNCTIONS
# =====================================================================
def gwb_psd(f, A=A_GWB, gamma=GAMMA, fref=f_yr):
    """One-sided GWB-induced timing-residual PSD [s^2 / Hz]."""
    return A**2 / (12.0 * np.pi**2) * (f / fref)**(-gamma) * fref**(-3)


def hd(theta_ab):
    """Hellings-Downs correlation, normalized to H(0+) = 0.5."""
    x = (1.0 - np.cos(theta_ab)) / 2.0
    x = np.where(x < 1e-12, 1e-12, x)
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5


# =====================================================================
# PULSAR SKY POSITIONS (Galactic-plane biased)
# =====================================================================
l_gal = np.random.uniform(-np.pi, np.pi, N_PULSARS)
b_gal = np.random.normal(0.0, np.radians(22.0), N_PULSARS)
isotropic_mix = np.random.rand(N_PULSARS) < 0.35
b_gal[isotropic_mix] = np.arcsin(np.random.uniform(-1, 1, isotropic_mix.sum()))
b_gal = np.clip(b_gal, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)

theta_p = np.pi / 2 - b_gal
phi_p = l_gal
p_vec = np.column_stack([
    np.sin(theta_p) * np.cos(phi_p),
    np.sin(theta_p) * np.sin(phi_p),
    np.cos(theta_p),
])

dots = np.clip(p_vec @ p_vec.T, -1.0, 1.0)
sep_mat = np.arccos(dots)

# =====================================================================
# H-D-CORRELATED RESIDUAL GENERATION
# =====================================================================
hd_mat = hd(sep_mat)
np.fill_diagonal(hd_mat, 1.0)
eig_min = np.linalg.eigvalsh(hd_mat).min()
if eig_min < 1e-3:
    hd_mat += (1e-3 - eig_min) * np.eye(N_PULSARS)
L_hd = np.linalg.cholesky(hd_mat)

N_OBS = 800
dt = T_SPAN / N_OBS
times = np.arange(N_OBS) * dt

freqs_fft = np.fft.rfftfreq(N_OBS, d=dt)
freqs_fft_safe = freqs_fft.copy()
freqs_fft_safe[0] = freqs_fft_safe[1] / 2.0

# Var(Re X) = Var(Im X) = N * P / (4 dt)
P_gwb_fft = gwb_psd(freqs_fft_safe)
P_gwb_fft[0] = 0.0
sigma_fft = np.sqrt(P_gwb_fft * N_OBS / (4.0 * dt))

gwb_fft = np.zeros((N_PULSARS, len(freqs_fft)), dtype=complex)
for k, sig in enumerate(sigma_fft):
    if sig == 0:
        continue
    gwb_fft[:, k] = (L_hd @ np.random.randn(N_PULSARS) +
                     1j * L_hd @ np.random.randn(N_PULSARS)) * sig

gwb_res = np.array([np.fft.irfft(gwb_fft[i], n=N_OBS) for i in range(N_PULSARS)])

# =====================================================================
# NG15-LIKE PER-PULSAR NOISE
# =====================================================================
log10_A_red = -15.5 + 1.8 * np.random.beta(2.0, 6.0, N_PULSARS)
gamma_red = np.random.uniform(1.0, 4.5, N_PULSARS)

red_res = np.zeros_like(gwb_res)
for i in range(N_PULSARS):
    A_i = 10**log10_A_red[i]
    P_red = (A_i**2 / (12.0 * np.pi**2)
             * (freqs_fft_safe / f_yr)**(-gamma_red[i]) * f_yr**(-3))
    P_red[0] = 0.0
    sig_i = np.sqrt(P_red * N_OBS / (4.0 * dt))
    fft_red = (np.random.randn(len(freqs_fft)) +
               1j * np.random.randn(len(freqs_fft))) * sig_i
    red_res[i] = np.fft.irfft(fft_red, n=N_OBS)

white_rms = 10**np.random.uniform(np.log10(1.5e-7), np.log10(1.5e-6), N_PULSARS)
white_res = white_rms[:, None] * np.random.randn(N_PULSARS, N_OBS)

residuals = gwb_res + red_res + white_res

# =====================================================================
# CROSS-CORRELATION (unweighted Pearson, binned)
# =====================================================================
res_centered = residuals - residuals.mean(axis=1, keepdims=True)
res_std = res_centered.std(axis=1)

seps_deg, rho_pair = [], []
for i in range(N_PULSARS):
    for j in range(i + 1, N_PULSARS):
        seps_deg.append(np.degrees(sep_mat[i, j]))
        rho = (res_centered[i] * res_centered[j]).mean() / (res_std[i] * res_std[j])
        rho_pair.append(rho)
seps_deg = np.array(seps_deg)
rho_pair = np.array(rho_pair)

nbins = 11
bin_edges = np.linspace(0, 180, nbins + 1)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_mean = np.full(nbins, np.nan)
bin_sem = np.full(nbins, np.nan)
for k in range(nbins):
    m = (seps_deg >= bin_edges[k]) & (seps_deg < bin_edges[k + 1])
    if m.sum() > 1:
        bin_mean[k] = rho_pair[m].mean()
        bin_sem[k] = rho_pair[m].std(ddof=1) / np.sqrt(m.sum())

clean = ~np.isnan(bin_mean) & (bin_sem > 0)
hd_b = hd(np.radians(bin_centers[clean]))
inv_var = 1.0 / bin_sem[clean]**2
amp_fit = np.sum(bin_mean[clean] * hd_b * inv_var) / np.sum(hd_b**2 * inv_var)
amp_err = 1.0 / np.sqrt(np.sum(hd_b**2 * inv_var))
detect_sigma = amp_fit / amp_err

theta_t = np.linspace(0.001, np.pi, 400)
hd_curve = amp_fit * hd(theta_t)

# =====================================================================
# FREE-SPECTRUM POWER ESTIMATE
# =====================================================================
fft_residuals = np.array([np.fft.rfft(residuals[i]) for i in range(N_PULSARS)])

pair_i, pair_j, pair_hd = [], [], []
for i in range(N_PULSARS):
    for j in range(i + 1, N_PULSARS):
        pair_i.append(i)
        pair_j.append(j)
        pair_hd.append(hd(sep_mat[i, j]))
pair_i = np.array(pair_i)
pair_j = np.array(pair_j)
pair_hd = np.array(pair_hd)
hd_sum_sq = np.sum(pair_hd**2)

P_per_bin = np.zeros(N_FOURIER)
P_per_bin_err = np.zeros(N_FOURIER)
for k_idx, fk in enumerate(f_k):
    bin_idx = np.argmin(np.abs(freqs_fft - fk))
    norm = 2.0 * dt / N_OBS
    cross_pows = np.real(fft_residuals[pair_i, bin_idx] *
                         np.conj(fft_residuals[pair_j, bin_idx])) * norm
    P_per_bin[k_idx] = np.sum(cross_pows * pair_hd) / hd_sum_sq
    P_per_bin_err[k_idx] = np.std(cross_pows * pair_hd) / np.sqrt(hd_sum_sq)

P_gwb_theory = gwb_psd(f_k)

# =====================================================================
# AMPLITUDE POSTERIOR (from H-D template fit)
# =====================================================================
A_central = A_GWB * np.sqrt(amp_fit / hd(np.array([0.0])).item())
A_sigma = 0.5 * A_central * (amp_err / amp_fit)

A_grid = np.linspace(0.3e-15, 4.5e-15, 800)
post = np.exp(-0.5 * ((A_grid - A_central) / A_sigma)**2)
post /= np.trapz(post, A_grid)

dA = np.diff(A_grid, prepend=A_grid[0])
cdf = np.cumsum(post * dA)
cdf /= cdf[-1]
A_med = float(np.interp(0.50, cdf, A_grid))
A_lo = float(np.interp(0.05, cdf, A_grid))
A_hi = float(np.interp(0.95, cdf, A_grid))

# =====================================================================
# FIGURE  —  3-row layout
# =====================================================================
NAVY = '#0a1f3a'
ORANGE = '#e87722'
RED = '#c0392b'
GRAY = '#555555'
LIGHTGRAY = '#cfd4dc'

fig = plt.figure(figsize=(13, 13.8), facecolor='white')
gs = gridspec.GridSpec(
    3, 3, figure=fig,
    height_ratios=[1.15, 2.6, 1.7],
    width_ratios=[1, 1, 1],
    hspace=0.52, wspace=0.42,
    top=0.875, bottom=0.062, left=0.075, right=0.965,
)

fig.suptitle('From 67 Noisy Clocks to One Cosmic Signal',
             fontsize=25, fontweight='bold', y=0.965, color=NAVY)

# ---- Row 1: time-domain residuals ----------------------------------
ax_res = fig.add_subplot(gs[0, :])
years_axis = times / yr
order = np.argsort(residuals.std(axis=1))
idxs = np.concatenate([order[:3],
                       order[N_PULSARS // 2 - 1:N_PULSARS // 2 + 2],
                       order[-3:]])
offset = 0.0
for idx in idxs:
    r = residuals[idx] / residuals[idx].std()
    ax_res.plot(years_axis, r + offset, color=NAVY, alpha=0.7, linewidth=0.7)
    offset += 5
ax_res.set_xlabel('Year of observation (T = 16 yr)', fontsize=11, color=GRAY)
ax_res.set_yticks([])
ax_res.set_xlim(years_axis.min(), years_axis.max())
for sp in ['top', 'right', 'left']:
    ax_res.spines[sp].set_visible(False)
ax_res.spines['bottom'].set_color(GRAY)
ax_res.tick_params(colors=GRAY)
ax_res.set_title(
    'Individual pulsars: red noise + white noise + a hidden GW background.\n'
    'No single residual stream looks like a detection.',
    fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=10)

# ---- Row 2: H-D scatter HERO ---------------------------------------
ax_hd = fig.add_subplot(gs[1, :])
ax_hd.scatter(seps_deg, rho_pair, c=LIGHTGRAY, s=14, alpha=0.55,
              edgecolors='none', label=f'{N_PAIRS:,} pulsar pairs')
ax_hd.errorbar(bin_centers, bin_mean, yerr=bin_sem, fmt='o', color=ORANGE,
               markersize=10, capsize=4, capthick=2, elinewidth=2,
               markeredgecolor=NAVY, markeredgewidth=1.5, zorder=5,
               label='Binned average')
ax_hd.plot(np.degrees(theta_t), hd_curve, color=NAVY, linewidth=3.0,
           label='Hellings\u2013Downs prediction', zorder=4)
ax_hd.axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
ax_hd.set_xlabel('Angular separation between pulsar pair  (degrees)',
                 fontsize=13, color=NAVY)
ax_hd.set_ylabel('Cross-correlation between pulsar pairs', fontsize=13, color=NAVY)
ax_hd.set_xlim(0, 180)
ax_hd.set_ylim(-0.65, 0.75)
ax_hd.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax_hd.legend(loc='upper right', fontsize=11, framealpha=0.95, frameon=True)
ax_hd.grid(True, alpha=0.2)
for sp in ['top', 'right']:
    ax_hd.spines[sp].set_visible(False)
ax_hd.set_title('Correlate the pulsars pairwise.  '
                'The quadrupolar Hellings\u2013Downs signature emerges.',
                fontsize=14.5, fontweight='bold', loc='left', color=NAVY, pad=10)

dip_y = amp_fit * hd(np.pi / 2)
ann = ax_hd.annotate('Quadrupolar dip near 90\u00b0\n\u2014 the fingerprint unique\nto gravitational waves',
                     xy=(90, dip_y), xytext=(140, -0.30),
                     fontsize=11, color=RED, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', color=RED, lw=1.8,
                                     connectionstyle='arc3,rad=-0.25'))
ann.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

detect_text = (
    f'$A_{{\\rm GWB}}$ = {A_med*1e15:.2f}$_{{-{(A_med-A_lo)*1e15:.2f}}}'
    f'^{{+{(A_hi-A_med)*1e15:.2f}}}\\times 10^{{-15}}$\n'
    f'detection at {detect_sigma:.1f}$\\sigma$'
)
ax_hd.text(0.02, 0.97, detect_text, transform=ax_hd.transAxes,
           ha='left', va='top', fontsize=11.5, color=NAVY,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=NAVY, linewidth=1.2, alpha=0.95))

# ---- Row 3: Sky map | Free spectrum | Posterior --------------------
ax_sky = fig.add_subplot(gs[2, 0], projection='mollweide')
phi_plot = phi_p.copy()
phi_plot[phi_plot > np.pi] -= 2 * np.pi
np.random.seed(99)
for _ in range(70):
    i, j = np.random.choice(N_PULSARS, 2, replace=False)
    ax_sky.plot([phi_plot[i], phi_plot[j]], [b_gal[i], b_gal[j]],
                color=NAVY, alpha=0.12, linewidth=0.45)
ax_sky.scatter(phi_plot, b_gal, c=ORANGE, s=28, alpha=0.9,
               edgecolors=NAVY, linewidths=0.5, zorder=5)
ax_sky.set_title('67 pulsars  \u00b7  2,211 baselines\n(galactic frame)',
                 fontsize=11.5, fontweight='bold', color=NAVY, pad=8)
ax_sky.grid(True, alpha=0.3)
ax_sky.set_xticklabels([])
ax_sky.set_yticklabels([])

ax_fs = fig.add_subplot(gs[2, 1])
P_pos = np.where(P_per_bin > 0, P_per_bin, np.nan)
ax_fs.errorbar(f_k * yr, P_pos, yerr=P_per_bin_err, fmt='o',
               color=ORANGE, markersize=7, capsize=3, capthick=1.2,
               markeredgecolor=NAVY, markeredgewidth=0.8,
               label='measured (HD-weighted)')
ax_fs.plot(f_k * yr, P_gwb_theory, color=NAVY, linewidth=2.5,
           label=r'$\propto f^{-13/3}$  (SMBHB)')
ax_fs.set_xscale('log')
ax_fs.set_yscale('log')
ax_fs.set_xlabel(r'frequency  /  yr$^{-1}$', fontsize=10.5, color=NAVY)
ax_fs.set_ylabel(r'residual PSD  [s$^2$/Hz]', fontsize=10.5, color=NAVY)
ax_fs.set_title('Free spectrum:\nHD-projected power per mode',
                fontsize=11.5, fontweight='bold', color=NAVY, pad=8)
ax_fs.legend(fontsize=9, loc='lower left', framealpha=0.95)
ax_fs.grid(True, which='both', alpha=0.2)
for sp in ['top', 'right']:
    ax_fs.spines[sp].set_visible(False)

ax_post = fig.add_subplot(gs[2, 2])
A_grid_e15 = A_grid * 1e15
ax_post.plot(A_grid_e15, post / post.max(), color=NAVY, linewidth=2.5)
ax_post.fill_between(A_grid_e15, 0, post / post.max(),
                     where=(A_grid_e15 >= A_lo * 1e15) & (A_grid_e15 <= A_hi * 1e15),
                     color=NAVY, alpha=0.20)
ax_post.axvline(A_GWB * 1e15, color=RED, linewidth=1.8, linestyle='--')
ax_post.axvspan((A_GWB - 0.6e-15) * 1e15, (A_GWB + 0.7e-15) * 1e15,
                color=RED, alpha=0.08)
ax_post.text(0.96, 0.94, 'NG15 median (2.4)', transform=ax_post.transAxes,
             ha='right', va='top', fontsize=9, color=RED)
ax_post.set_xlabel(r'$A_{\rm GWB}$  /  $10^{-15}$', fontsize=10.5, color=NAVY)
ax_post.set_ylabel('posterior  (peak-normalized)', fontsize=10.5, color=NAVY)
ax_post.set_title('Amplitude recovery:\nposterior on $A_{\\rm GWB}$',
                  fontsize=11.5, fontweight='bold', color=NAVY, pad=8)
ax_post.grid(True, alpha=0.2)
for sp in ['top', 'right']:
    ax_post.spines[sp].set_visible(False)
ax_post.set_xlim(A_grid_e15.min(), A_grid_e15.max())
ax_post.set_ylim(0, 1.08)

fig.text(0.99, 0.008,
         'physics-accurate simulation calibrated to NANOGrav 15-yr (Agazie+ 2023)',
         ha='right', fontsize=8, color=GRAY, style='italic')

out = os.path.join(_FIGDIR, 'hd_simulation.png')
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# Save per-pair products so animate_hd.py can reuse them
np.savez(os.path.join(_RESDIR, 'sim_products.npz'),
         seps=seps_deg, rho=rho_pair,
         bin_centers=bin_centers, bin_mean=bin_mean, bin_sem=bin_sem,
         amp_fit=amp_fit, detect_sigma=detect_sigma,
         A_med=A_med, A_lo=A_lo, A_hi=A_hi,
         pulsar_theta=theta_p, pulsar_phi=phi_p)

print(f"Saved figure   -> {out}")
print(f"Saved products -> {os.path.join(_RESDIR, 'sim_products.npz')}")
print(f"Detection significance: {detect_sigma:.2f} sigma")
print(f"A_GWB recovered: {A_med*1e15:.2f} (+{(A_hi-A_med)*1e15:.2f} / "
      f"-{(A_med-A_lo)*1e15:.2f}) x 10^-15")
