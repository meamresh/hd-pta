"""
ng15_optimal_statistic.py
=========================
Genuine NANOGrav 15-year Hellings-Downs measurement.

Runs the official `enterprise` optimal-statistic pipeline on the REAL
NANOGrav 15-year data plus the real presampled red-noise MCMC chain.
This is not a simulation. Inputs:

  * real NG15 pulsars        tutorials/data/feathers/  (67 .feather files)
  * real white-noise params  tutorials/data/15yr_wn_dict.json
  * real red-noise chain     presampled_cores/curn_14f_pl_vg.core
  * the collaboration's own  enterprise_extensions OptimalStatistic

all of which ship inside the NANOGrav repo cloned by setup.sh.

OUTPUT
------
results/ng15_real_os.npz  -- per-pair angular separations, noise-
marginalized correlations + uncertainties, fitted A^2, and HD S/N.
The animation script consumes this file.

MEMORY
------
The official model_2a on all 67 pulsars needs ~4-5 GB RAM. This script
auto-detects available memory and selects the largest pulsar subset that
fits. On an 8 GB+ machine (or Google Colab) it uses all 67 pulsars and
reproduces the full 2,211-pair result; on a small machine it falls back
to a subset (fewer pairs, lower S/N) and says so loudly.

Run:  python src/ng15_optimal_statistic.py [--n-pulsars N] [--n-noise M]
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, 'external', '15yr_stochastic_analysis')
DATADIR = os.path.join(REPO, 'tutorials', 'data')
CHAIN = os.path.join(REPO, 'tutorials', 'presampled_cores',
                     'curn_14f_pl_vg.core')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)


def available_ram_gb():
    """Best-effort free-RAM estimate, in GB."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError):
        pass
    return 8.0  # optimistic default if /proc unavailable (e.g. macOS)


def choose_n_pulsars(ram_gb):
    """Largest pulsar count expected to fit in the given RAM."""
    if ram_gb >= 7.5:
        return 67          # full array
    if ram_gb >= 5.0:
        return 45
    if ram_gb >= 3.5:
        return 30
    return 22              # minimum useful subset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-pulsars', type=int, default=None,
                    help='number of pulsars to use (default: auto from RAM)')
    ap.add_argument('--n-noise', type=int, default=None,
                    help='noise realizations to marginalize over '
                         '(default: 10000 if full array else 200)')
    args = ap.parse_args()

    if not os.path.isdir(DATADIR):
        sys.exit('ERROR: NANOGrav data not found. Run ./setup.sh first.')

    # ---- imports deferred so --help works without the heavy stack ----
    from enterprise_extensions import models
    from enterprise_extensions.frequentist import optimal_statistic as opt_stat
    from enterprise_extensions.load_feathers import load_feathers_from_folder
    from la_forge import core

    ram = available_ram_gb()
    n_psr = args.n_pulsars or choose_n_pulsars(ram)
    n_noise = args.n_noise or (10000 if n_psr >= 67 else 200)

    print('=' * 62)
    print('NANOGrav 15-year optimal statistic  --  REAL DATA')
    print('=' * 62)
    print(f'  available RAM (est.) : {ram:.1f} GB')
    print(f'  pulsars to use       : {n_psr}')
    print(f'  noise realizations   : {n_noise}')
    if n_psr < 67:
        print(f'  NOTE: using a {n_psr}-pulsar subset to fit in RAM.')
        print(f'        The full 67-pulsar array (2,211 pairs) reproduces')
        print(f'        the published 3.5-4 sigma detection; a subset has')
        print(f'        fewer pairs and lower S/N. Use a >=8 GB machine')
        print(f'        (or Google Colab) for the full result.')
    print()

    print('Loading real NANOGrav 15-year pulsars (feather files)...')
    psrs_all = load_feathers_from_folder(os.path.join(DATADIR, 'feathers'))
    print(f'  {len(psrs_all)} pulsars available in the release')

    # subset = pulsars with fewest TOAs (purely a memory criterion)
    psrs = sorted(psrs_all, key=lambda p: len(p.toas))[:n_psr]
    n_pairs = n_psr * (n_psr - 1) // 2
    print(f'  using {n_psr} pulsars  ->  {n_pairs} pulsar pairs')

    print('Loading real white-noise parameters...')
    with open(os.path.join(DATADIR, '15yr_wn_dict.json')) as fh:
        wn_params = json.load(fh)

    print('Building model_2a PTA  (a few minutes, several GB RAM)...')
    pta = models.model_2a(psrs, noisedict=wn_params,
                          n_gwbfreqs=14, tm_svd=True)

    print('Constructing OptimalStatistic...')
    os_obj = opt_stat.OptimalStatistic(psrs, bayesephem=False,
                                       noisedict=wn_params, pta=pta)

    print('Loading real red-noise posterior chain...')
    curn = core.Core(corepath=CHAIN)

    print(f'Running noise-marginalized optimal statistic '
          f'(N={n_noise})...')
    xi, rho, sig, A, SNR = os_obj.compute_noise_marginalized_os(
        curn.chain, curn.params, N=n_noise)

    xi = np.asarray(xi)
    rho = np.asarray(rho)
    sig = np.asarray(sig)
    A = np.asarray(A)
    SNR = np.asarray(SNR)

    print()
    print('=' * 62)
    print('RESULT')
    print('=' * 62)
    print(f'  pulsar pairs        : {len(xi)}')
    print(f'  HD signal-to-noise  : {np.median(SNR):.2f} '
          f'(median over {n_noise} draws)')
    print(f'  A^2 (GWB)           : {np.median(A):.3e}')
    print(f'  A_GWB               : {np.sqrt(np.median(A)):.3e}')

    out = os.path.join(RESULTS, 'ng15_real_os.npz')
    np.savez(out,
             xi=xi, rho=rho, sig=sig, A=A, SNR=SNR,
             rho_mean=rho.mean(axis=0), rho_err=rho.std(axis=0),
             psr_names=np.array([p.name for p in psrs]),
             psr_pos=np.array([p.pos for p in psrs]),
             n_pulsars=n_psr, n_noise=n_noise)
    print(f'\nSaved -> {out}')
    print('Next:  python src/animate_hd.py --real')


if __name__ == '__main__':
    main()
