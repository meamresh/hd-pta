#!/usr/bin/env bash
# ---------------------------------------------------------------------
# setup.sh -- fetch the real NANOGrav 15-year data
#
# Clones the NANOGrav collaboration's public analysis repository, which
# ships the real reduced 15-year dataset (67 pulsars in feather format),
# the real white-noise parameters, and the real presampled red-noise
# MCMC chain. Total download is a few hundred MB.
#
# Only needed for the real-data pipeline (src/ng15_optimal_statistic.py).
# The simulation pipeline (src/simulate_hd.py) needs nothing external.
# ---------------------------------------------------------------------
set -e

EXTERNAL_DIR="external"
REPO_URL="https://github.com/nanograv/15yr_stochastic_analysis.git"
TARGET="${EXTERNAL_DIR}/15yr_stochastic_analysis"

mkdir -p "${EXTERNAL_DIR}"

if [ -d "${TARGET}" ]; then
    echo "NANOGrav data repo already present at ${TARGET}"
else
    echo "Cloning NANOGrav 15-year analysis repository..."
    git clone --depth 1 "${REPO_URL}" "${TARGET}"
fi

echo ""
echo "Verifying real data files..."
FEATHERS="${TARGET}/tutorials/data/feathers"
N_FEATHER=$(ls "${FEATHERS}"/*.feather 2>/dev/null | wc -l | tr -d ' ')
echo "  pulsar feather files : ${N_FEATHER}  (expected 67)"

CHAIN="${TARGET}/tutorials/presampled_cores/curn_14f_pl_vg.core"
if [ -f "${CHAIN}" ]; then
    echo "  red-noise MCMC chain : present"
else
    echo "  red-noise MCMC chain : MISSING"
fi

WN="${TARGET}/tutorials/data/15yr_wn_dict.json"
if [ -f "${WN}" ]; then
    echo "  white-noise params   : present"
else
    echo "  white-noise params   : MISSING"
fi

echo ""
echo "Setup complete. Next:"
echo "  python src/simulate_hd.py            # simulation (no data needed)"
echo "  python src/ng15_optimal_statistic.py # real NANOGrav measurement"
