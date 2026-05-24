"""
animate_npsr.py
===============
Animate the Hellings-Downs curve sharpening as the number of pulsars in
the array grows from a handful to all 67.

Why this view is informative
----------------------------
The pair-by-pair animation (animate_hd.py) reveals correlations in a
random order; it shows the curve filling in. This script does something
different: it picks the first N pulsars, forms all N*(N-1)/2 pairs from
that subset, and steps N from ~6 up to the full array. The signal-to-
noise of the Hellings-Downs detection grows roughly as the number of
pairs ~ N^2, so the curve visibly sharpens, the dip at 90 deg deepens
to its theoretical value, and the binned error bars shrink frame by
frame.

Two modes:
  --sim   (default)  use simulation products from simulate_hd.py
  --real             use real NANOGrav products from
                     ng15_optimal_statistic.py

Run:  python src/animate_npsr.py [--sim | --real]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_FIGDIR = os.path.join(_ROOT, 'figures')
_RESDIR = os.path.join(_ROOT, 'results')

NAVY = '#0a1f3a'
ORANGE = '#e87722'
RED = '#c0392b'
GRAY = '#555555'
LIGHTGRAY = '#cfd4dc'


def hd(theta_rad):
    """Hellings-Downs correlation, normalized to H(0+) = 0.5."""
    x = (1.0 - np.cos(theta_rad)) / 2.0
    x = np.where(x < 1e-12, 1e-12, x)
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5


# ---------------------------------------------------------------------
# Load products
# ---------------------------------------------------------------------
def load_sim():
    """Simulation products from simulate_hd.py.

    The simulation script computes per-pair (sep, rho) over ALL pulsars
    in upper-triangular order. To restrict to the first N pulsars we
    recompute pair indices and select.
    """
    path = os.path.join(_RESDIR, 'sim_products.npz')
    if not os.path.exists(path):
        raise SystemExit('Run `python src/simulate_hd.py` first.')
    d = np.load(path)
    seps = d['seps']
    rho = d['rho']
    theta_p = d['pulsar_theta']
    phi_p = d['pulsar_phi']

    n_psr = len(theta_p)
    lon = phi_p.copy()
    lon[lon > np.pi] -= 2 * np.pi
    lat = np.pi / 2 - theta_p

    # uniform error proxy (sim has no per-pair sigma)
    err = np.full_like(rho, max(rho.std(), 1e-3))

    return dict(seps=seps, rho=rho, err=err, lon=lon, lat=lat,
                n_psr=n_psr,
                ylabel='Cross-correlation',
                ylim=(-0.75, 0.85),
                title='Simulation calibrated to NANOGrav 15-yr',
                hd_normalized=False)


def load_real():
    """Real-data products from ng15_optimal_statistic.py.

    The OS returns rho ~ A^2 * Gamma_HD. We divide by the fitted A^2
    so the points sit on the dimensionless H-D curve.
    """
    path = os.path.join(_RESDIR, 'ng15_real_os.npz')
    if not os.path.exists(path):
        raise SystemExit('Run `python src/ng15_optimal_statistic.py` first.')
    d = np.load(path, allow_pickle=True)
    seps = np.degrees(np.asarray(d['xi']))
    rho = np.asarray(d['rho_mean'])
    err = np.asarray(d['rho_err'])
    A2 = np.asarray(d['A'])
    pos = np.asarray(d['psr_pos'])

    n_psr = len(pos)
    A2_med = np.median(A2)
    rho = rho / A2_med
    err = err / A2_med
    lon = np.arctan2(pos[:, 1], pos[:, 0])
    lat = np.arcsin(np.clip(pos[:, 2], -1, 1))

    return dict(seps=seps, rho=rho, err=err, lon=lon, lat=lat,
                n_psr=n_psr,
                ylabel='Correlation  /  fitted GWB amplitude$^2$',
                ylim=(-1.8, 2.1),
                title='Real NANOGrav 15-yr data (optimal statistic)',
                hd_normalized=True)


# ---------------------------------------------------------------------
# Pair index helper
# ---------------------------------------------------------------------
def pair_indices(n_psr):
    """Pulsar indices for each pair in upper-triangular order."""
    pi, pj = [], []
    for i in range(n_psr):
        for j in range(i + 1, n_psr):
            pi.append(i)
            pj.append(j)
    return np.array(pi), np.array(pj)


# ---------------------------------------------------------------------
# Build animation
# ---------------------------------------------------------------------
def build(mode):
    P = load_sim() if mode == 'sim' else load_real()
    seps_all, rho_all, err_all = P['seps'], P['rho'], P['err']
    lon, lat, n_psr = P['lon'], P['lat'], P['n_psr']

    pair_i_all, pair_j_all = pair_indices(n_psr)
    assert len(pair_i_all) == len(seps_all), \
        f'pair-count mismatch: {len(pair_i_all)} vs {len(seps_all)}'

    # Frame schedule: N grows from a small floor to n_psr, more frames
    # near the start where each added pulsar matters most.
    N_FRAMES_GROW = 50
    N_HOLD = 14
    # logarithmic-ish growth + linear tail looks better than pure linear
    n_floor = 6
    grow = np.unique(np.round(np.geomspace(n_floor, n_psr,
                                           N_FRAMES_GROW)).astype(int))
    # pad if uniqueness shrank the count
    while len(grow) < N_FRAMES_GROW:
        grow = np.append(grow, n_psr)
    n_schedule = np.concatenate([grow, np.full(N_HOLD, n_psr)])

    nbins = 11
    edges = np.linspace(0, 180, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    theta_t = np.linspace(0.001, np.pi, 400)
    hd_t = hd(theta_t)

    # ---- figure ----
    fig = plt.figure(figsize=(12.5, 5.85), facecolor='white')
    fig.subplots_adjust(left=0.04, right=0.975, top=0.78, bottom=0.13,
                        wspace=0.20)
    ax_sky = fig.add_subplot(1, 2, 1, projection='mollweide')
    ax_hd = fig.add_subplot(1, 2, 2)

    fig.suptitle(f'{P["title"]}: more pulsars, sharper signal',
                 fontsize=14.5, fontweight='bold', color=NAVY, y=0.955)
    status = fig.text(0.5, 0.875, '', ha='center', va='center',
                      fontsize=11, color=GRAY)

    # static sky-map background: all pulsars in light grey (target set)
    ax_sky.scatter(lon, lat, c=LIGHTGRAY, s=22, alpha=0.85,
                   edgecolors=GRAY, linewidths=0.4, zorder=3)
    # active pulsars overlay (updated each frame)
    active_scat = ax_sky.scatter([], [], c=ORANGE, s=42, alpha=0.95,
                                 edgecolors=NAVY, linewidths=0.6, zorder=6)
    ax_sky.set_title('Pulsars currently in the array',
                     fontsize=11, fontweight='bold', color=NAVY, pad=10)
    ax_sky.grid(True, alpha=0.3)
    ax_sky.set_xticklabels([])
    ax_sky.set_yticklabels([])

    # H-D panel static
    ax_hd.axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    ax_hd.set_xlim(0, 180)
    ax_hd.set_ylim(*P['ylim'])
    ax_hd.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax_hd.set_xlabel('Angular separation between pulsar pair  (degrees)',
                     fontsize=11, color=NAVY)
    ax_hd.set_ylabel(P['ylabel'], fontsize=11, color=NAVY)
    ax_hd.set_title('Pairwise correlation vs. separation',
                    fontsize=11, fontweight='bold', color=NAVY, pad=10)
    ax_hd.grid(True, alpha=0.2)
    for sp in ['top', 'right']:
        ax_hd.spines[sp].set_visible(False)

    # reference (target) H-D curve - faint dashed, what we are converging to
    if P['hd_normalized']:
        # real data: theoretical H-D shape, amplitude 1
        ax_hd.plot(np.degrees(theta_t), hd_t, color=NAVY, linewidth=1.3,
                   linestyle='--', alpha=0.4, zorder=3,
                   label='Hellings\u2013Downs target')
    else:
        # simulation: the asymptotic fit using all pairs
        from numpy import sum as _sum
        bm_all = np.full(nbins, np.nan)
        bs_all = np.full(nbins, np.nan)
        for k in range(nbins):
            m = (seps_all >= edges[k]) & (seps_all < edges[k + 1])
            if m.sum() > 2:
                bm_all[k] = rho_all[m].mean()
                bs_all[k] = rho_all[m].std(ddof=1) / np.sqrt(m.sum())
        clean = ~np.isnan(bm_all) & (bs_all > 0)
        hd_b = hd(np.radians(centers[clean]))
        iv = 1.0 / bs_all[clean]**2
        amp_target = _sum(bm_all[clean] * hd_b * iv) / _sum(hd_b**2 * iv)
        ax_hd.plot(np.degrees(theta_t), amp_target * hd_t,
                   color=NAVY, linewidth=1.3, linestyle='--', alpha=0.4,
                   zorder=3, label='Final-array target')

    ax_hd.legend(loc='lower right', fontsize=9, framealpha=0.95)

    # dynamic artists
    scatter = ax_hd.scatter([], [], c=LIGHTGRAY, s=15, alpha=0.5,
                            edgecolors='none', zorder=2)
    binned = ax_hd.errorbar([], [], yerr=[], fmt='o', color=ORANGE,
                            markersize=8, capsize=3, capthick=1.6,
                            elinewidth=1.6, markeredgecolor=NAVY,
                            markeredgewidth=1.2, zorder=5)
    binned_line, _, binned_bars = binned
    (curve,) = ax_hd.plot([], [], color=NAVY, linewidth=2.8, zorder=4)
    info_txt = ax_hd.text(0.03, 0.95, '', transform=ax_hd.transAxes,
                          ha='left', va='top', fontsize=10.5, color=NAVY,
                          bbox=dict(boxstyle='round,pad=0.45',
                                    facecolor='white', edgecolor=NAVY,
                                    linewidth=1.1, alpha=0.95))

    def update(frame):
        N = int(n_schedule[frame])
        # mask of pairs whose BOTH pulsars are in the first N
        m_pair = (pair_i_all < N) & (pair_j_all < N)
        s_n = seps_all[m_pair]
        r_n = rho_all[m_pair]
        e_n = err_all[m_pair]
        n_pairs = len(s_n)

        # update active pulsar overlay
        active_scat.set_offsets(np.column_stack([lon[:N], lat[:N]]))

        # all pair points in the H-D panel (faint scatter)
        scatter.set_offsets(np.column_stack([s_n, r_n]))

        # binned: compute both fit uncertainty (sem) and a more conservative
        # plotted spread (within-bin scatter). Use the fit sem for the
        # template weights and the plotted spread (halved) for visual bars.
        bm, bs_fit, bs_plot, bc = [], [], [], []
        for k in range(nbins):
            mk = (s_n >= edges[k]) & (s_n < edges[k + 1])
            if mk.sum() >= 3:
                vals = r_n[mk]
                if mode == 'sim':
                    mean = vals.mean()
                    fit_sem = vals.std(ddof=1) / np.sqrt(mk.sum())
                    plot_sem = vals.std(ddof=1)
                else:
                    w = 1.0 / e_n[mk]**2
                    mean = np.sum(vals * w) / np.sum(w)
                    fit_sem = 1.0 / np.sqrt(np.sum(w))
                    plot_sem = np.sqrt(np.average((vals - mean)**2,
                                                  weights=w))
                # avoid unrealistically tiny fit uncertainties by flooring
                # the fit sem to at least half the within-bin scatter
                bm.append(mean)
                bs_plot.append(plot_sem)
                bs_fit.append(max(fit_sem, 0.12 * plot_sem, 1e-3))
                bc.append(centers[k])
        bm, bs_fit, bs_plot, bc = (
            np.array(bm), np.array(bs_fit), np.array(bs_plot), np.array(bc)
        )

        if len(bm) > 0:
            binned_line.set_data(bc, bm)
            binned_bars[0].set_segments(
                [np.array([[x, y - e], [x, y + e]])
                 for x, y, e in zip(bc, bm, 0.5 * bs_plot)])
        else:
            binned_line.set_data([], [])
            binned_bars[0].set_segments([])

        # live template fit
        sig_text = '\u2014'
        if len(bm) >= 4:
            iv = 1.0 / bs_fit**2
            hd_b = hd(np.radians(bc))
            amp = np.sum(bm * hd_b * iv) / np.sum(hd_b**2 * iv)
            amp_err = 1.0 / np.sqrt(np.sum(hd_b**2 * iv))
            sig = amp / amp_err if amp_err > 0 else 0.0
            curve.set_data(np.degrees(theta_t), amp * hd_t)
            sig_text = f'{max(sig,0):.1f}$\\sigma$'
        else:
            curve.set_data([], [])

        info_txt.set_text(f'pulsars: {N} / {n_psr}\n'
                          f'pairs:   {n_pairs:,}\n'
                          f'detection: {sig_text}')

        if frame < len(n_schedule) - N_HOLD:
            status.set_text(f'Growing the array \u2014 {N} pulsars '
                            f'\u2192 {n_pairs:,} pairs')
        else:
            status.set_text(f'Full array: {n_psr} pulsars, {n_pairs:,} pairs')

        return (scatter, binned_line, curve, info_txt, status,
                active_scat)

    anim = FuncAnimation(fig, update, frames=len(n_schedule),
                         interval=110, blit=False)
    os.makedirs(_FIGDIR, exist_ok=True)
    out = os.path.join(_FIGDIR, f'hd_growing_array_{mode}.gif')
    anim.save(out, writer=PillowWriter(fps=10), dpi=95)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out)/1e6:.1f} MB, '
          f'{len(n_schedule)} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sim', action='store_true', help='animate simulation (default)')
    g.add_argument('--real', action='store_true', help='animate real NANOGrav data')
    args = ap.parse_args()
    build('real' if args.real else 'sim')
