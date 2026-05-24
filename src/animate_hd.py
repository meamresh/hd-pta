"""
animate_hd.py
=============
Animated Hellings-Downs emergence: pulsar pairs are correlated one batch
at a time and the quadrupolar H-D signature is fitted live.

Two modes:

  --sim    (default)  Use simulation products from simulate_hd.py.
                      Clean H-D curve emerges; pedagogically clear.
                      Run `python src/simulate_hd.py` first.

  --real              Use real-data products from
                      ng15_optimal_statistic.py (genuine NANOGrav
                      optimal statistic). Honest measurement; the curve
                      is only clearly visible with the full 67-pulsar
                      array. Run `python src/ng15_optimal_statistic.py`
                      first.

Output: figures/hd_emergence_{sim,real}.gif

Run:  python src/animate_hd.py [--sim | --real]
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
# Load products for the requested mode
# ---------------------------------------------------------------------
def load_sim():
    path = os.path.join(_RESDIR, 'sim_products.npz')
    if not os.path.exists(path):
        raise SystemExit('Run `python src/simulate_hd.py` first.')
    d = np.load(path)
    seps = d['seps']
    rho = d['rho']
    # pulsar sky positions
    theta_p = d['pulsar_theta']
    phi_p = d['pulsar_phi']
    lon = phi_p.copy()
    lon[lon > np.pi] -= 2 * np.pi
    lat = np.pi / 2 - theta_p
    n_psr = len(theta_p)
    # uniform per-pair error proxy for binning (sim has no per-pair sigma)
    err = np.full_like(rho, rho.std())
    return dict(seps=seps, rho=rho, err=err, lon=lon, lat=lat,
                n_psr=n_psr, ylabel='Cross-correlation',
                ylim=(-0.65, 0.78), title='Simulation calibrated to '
                'NANOGrav 15-yr', subtitle_run='Adding pulsar pairs',
                detect_label=None)


def load_real():
    path = os.path.join(_RESDIR, 'ng15_real_os.npz')
    if not os.path.exists(path):
        raise SystemExit('Run `python src/ng15_optimal_statistic.py` first.')
    d = np.load(path, allow_pickle=True)
    seps = np.degrees(np.asarray(d['xi']))
    rho = np.asarray(d['rho_mean'])
    err = np.asarray(d['rho_err'])
    A2 = np.asarray(d['A'])
    SNR = np.asarray(d['SNR'])
    pos = np.asarray(d['psr_pos'])
    n_psr = len(pos)
    # OS normalization: rho ~ A^2 * Gamma_HD -> divide by A^2
    A2_med = np.median(A2)
    rho = rho / A2_med
    err = err / A2_med
    lon = np.arctan2(pos[:, 1], pos[:, 0])
    lat = np.arcsin(np.clip(pos[:, 2], -1, 1))
    return dict(seps=seps, rho=rho, err=err, lon=lon, lat=lat,
                n_psr=n_psr,
                ylabel='Correlation  /  fitted GWB amplitude$^2$',
                ylim=(-1.6, 1.9),
                title='Real NANOGrav 15-yr data (optimal statistic)',
                subtitle_run='Correlating real NANOGrav pulsar pairs',
                detect_label=f'H\u2013D S/N: {np.median(SNR):.1f}')


# ---------------------------------------------------------------------
# Build animation
# ---------------------------------------------------------------------
def build(mode):
    P = load_sim() if mode == 'sim' else load_real()
    seps, rho, err = P['seps'], P['rho'], P['err']
    lon, lat, n_psr = P['lon'], P['lat'], P['n_psr']
    n_pairs = len(seps)

    # pair -> pulsar index map (upper-triangular order)
    pi_idx, pj_idx = [], []
    for i in range(n_psr):
        for j in range(i + 1, n_psr):
            pi_idx.append(i)
            pj_idx.append(j)
    pi_idx = np.array(pi_idx[:n_pairs])
    pj_idx = np.array(pj_idx[:n_pairs])

    # shuffle reveal order
    rng = np.random.default_rng(15)
    sh = rng.permutation(n_pairs)
    seps_s, rho_s, err_s = seps[sh], rho[sh], err[sh]
    pi_s, pj_s = pi_idx[sh], pj_idx[sh]

    n_reveal, n_hold = 52, 16
    n_frames = n_reveal + n_hold
    start = max(25, n_pairs // 40)
    reveal_counts = np.linspace(start, n_pairs, n_reveal).astype(int)

    nbins = 11
    edges = np.linspace(0, 180, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    theta_t = np.linspace(0.001, np.pi, 400)
    hd_t = hd(theta_t)

    fig = plt.figure(figsize=(12.5, 5.7), facecolor='white')
    fig.subplots_adjust(left=0.04, right=0.975, top=0.79, bottom=0.135,
                        wspace=0.18)
    ax_sky = fig.add_subplot(1, 2, 1, projection='mollweide')
    ax_hd = fig.add_subplot(1, 2, 2)

    fig.suptitle(f'{P["title"]}: the Hellings\u2013Downs Signal Emerges',
                 fontsize=14.5, fontweight='bold', color=NAVY, y=0.965)
    status = fig.text(0.5, 0.875, '', ha='center', va='center',
                      fontsize=11, color=GRAY)

    ax_sky.scatter(lon, lat, c=ORANGE, s=28, alpha=0.95,
                   edgecolors=NAVY, linewidths=0.6, zorder=5)
    ax_sky.set_title(f'{n_psr} pulsars', fontsize=11,
                     fontweight='bold', color=NAVY, pad=10)
    ax_sky.grid(True, alpha=0.3)
    ax_sky.set_xticklabels([])
    ax_sky.set_yticklabels([])

    ax_hd.axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
    ax_hd.set_xlim(0, 180)
    ax_hd.set_ylim(*P['ylim'])
    ax_hd.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax_hd.set_xlabel('Angular separation between pulsar pair  (degrees)',
                     fontsize=11, color=NAVY)
    ax_hd.set_ylabel(P['ylabel'], fontsize=11, color=NAVY)
    ax_hd.set_title('Pairwise correlation vs. separation', fontsize=11,
                    fontweight='bold', color=NAVY, pad=10)
    ax_hd.grid(True, alpha=0.2)
    for sp in ['top', 'right']:
        ax_hd.spines[sp].set_visible(False)

    # reference H-D shape (real mode only - the prediction to compare to)
    if mode == 'real':
        ax_hd.plot(np.degrees(theta_t), hd_t, color=NAVY, linewidth=1.3,
                   linestyle='--', alpha=0.45, zorder=3)

    state = {'sky_lines': []}
    scatter = ax_hd.scatter([], [], c=LIGHTGRAY, s=16, alpha=0.58,
                            edgecolors='none', zorder=2)
    binned = ax_hd.errorbar([], [], yerr=[], fmt='o', color=ORANGE,
                            markersize=8, capsize=3, capthick=1.6,
                            elinewidth=1.6, markeredgecolor=NAVY,
                            markeredgewidth=1.2, zorder=5)
    binned_line, _, binned_bars = binned
    (curve,) = ax_hd.plot([], [], color=NAVY, linewidth=2.8, zorder=4)
    detect_txt = ax_hd.text(0.03, 0.95, '', transform=ax_hd.transAxes,
                            ha='left', va='top', fontsize=10.5, color=NAVY,
                            bbox=dict(boxstyle='round,pad=0.45',
                                      facecolor='white', edgecolor=NAVY,
                                      linewidth=1.1, alpha=0.95))
    dip_ann = ax_hd.annotate('', xy=(90, 0),
                             xytext=(138, P['ylim'][0] * 0.62),
                             fontsize=10, color=RED, fontweight='bold',
                             ha='center',
                             arrowprops=dict(arrowstyle='->', color=RED,
                                             lw=1.6,
                                             connectionstyle='arc3,rad=-0.25'))
    dip_ann.set_path_effects([pe.withStroke(linewidth=3,
                                            foreground='white')])
    dip_ann.set_visible(False)

    def update(frame):
        n = reveal_counts[frame] if frame < n_reveal else n_pairs
        scatter.set_offsets(np.column_stack([seps_s[:n], rho_s[:n]]))

        for ln in state['sky_lines']:
            ln.remove()
        state['sky_lines'] = []
        if frame < n_reveal:
            b0 = max(0, n - 22)
            for idx in range(b0, n):
                a, b = pi_s[idx], pj_s[idx]
                ln, = ax_sky.plot([lon[a], lon[b]], [lat[a], lat[b]],
                                  color=ORANGE, alpha=0.32,
                                  linewidth=0.7, zorder=3)
                state['sky_lines'].append(ln)

        # binned points: keep the fit weights tied to the mean uncertainty,
        # but draw a more conservative within-bin spread on the plot.
        bm, bs_fit, bs_plot, bc = [], [], [], []
        for k in range(nbins):
            m = (seps_s[:n] >= edges[k]) & (seps_s[:n] < edges[k + 1])
            if m.sum() >= 3:
                vals = rho_s[:n][m]
                if mode == 'sim':
                    mean = vals.mean()
                    fit_sem = vals.std(ddof=1) / np.sqrt(m.sum())
                    plot_sem = vals.std(ddof=1)
                else:
                    w = 1.0 / err_s[:n][m]**2
                    mean = np.sum(vals * w) / np.sum(w)
                    fit_sem = 1.0 / np.sqrt(np.sum(w))
                    plot_sem = np.sqrt(np.average((vals - mean)**2,
                                                  weights=w))
                bm.append(mean)
                bs_fit.append(fit_sem)
                bs_plot.append(plot_sem)
                bc.append(centers[k])
        bm, bs_fit, bs_plot, bc = (
            np.array(bm), np.array(bs_fit), np.array(bs_plot), np.array(bc)
        )

        if len(bm) > 0:
            binned_line.set_data(bc, bm)
            binned_bars[0].set_segments(
                [np.array([[x, y - e], [x, y + e]])
                 for x, y, e in zip(bc, bm, 0.5*bs_plot)])
        else:
            binned_line.set_data([], [])
            binned_bars[0].set_segments([])

        # live H-D template fit
        if len(bm) >= 4 and np.all(bs_fit > 0):
            iv = 1.0 / bs_fit**2
            hd_b = hd(np.radians(bc))
            amp = np.sum(bm * hd_b * iv) / np.sum(hd_b**2 * iv)
            curve.set_data(np.degrees(theta_t), amp * hd_t)
            if mode == 'sim':
                amp_err = 1.0 / np.sqrt(np.sum(hd_b**2 * iv))
                sig = amp / amp_err if amp_err > 0 else 0.0
                detect_txt.set_text(f'pairs: {n:,} / {n_pairs:,}\n'
                                    f'detection: {max(sig,0):.1f}$\\sigma$')
                if sig > 4:
                    dip_ann.set_visible(True)
                    dip_ann.set_text('Quadrupolar dip\nnear 90\u00b0')
                    dip_ann.xy = (90, amp * hd(np.pi / 2))
                else:
                    dip_ann.set_visible(False)
            else:
                detect_txt.set_text(f'real pairs: {n} / {n_pairs}\n'
                                    f'{P["detect_label"]}')
        else:
            curve.set_data([], [])
            detect_txt.set_text(f'pairs: {n:,} / {n_pairs:,}\n'
                                'detection: \u2014')
            dip_ann.set_visible(False)

        if frame < n_reveal:
            status.set_text(P['subtitle_run'] +
                            ' \u2014 each point is one pulsar-pair correlation')
        else:
            status.set_text(f'All {n_pairs:,} pairs correlated')

        return scatter, binned_line, curve, detect_txt, dip_ann, status

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=90, blit=False)
    out = os.path.join(_FIGDIR, f'hd_emergence_{mode}.gif')
    os.makedirs(_FIGDIR, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=12), dpi=95)
    plt.close()
    size = os.path.getsize(out) / 1e6
    print(f'Saved: {out}  ({size:.1f} MB, {n_frames} frames)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sim', action='store_true',
                   help='animate the simulation (default)')
    g.add_argument('--real', action='store_true',
                   help='animate the real NANOGrav optimal-statistic result')
    args = ap.parse_args()
    build('real' if args.real else 'sim')
