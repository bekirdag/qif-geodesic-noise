#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data/BBH_snr_306}"
DEVICE="${DEVICE:-auto}"
MAX_SECONDS="${MAX_SECONDS:-128}"
MAX_BINS="${MAX_BINS:-128}"
MAX_ITER="${MAX_ITER:-120}"
N_STARTS="${N_STARTS:-2}"
FIT_PHI="${FIT_PHI:-1}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="$ROOT_DIR/test-runs/${STAMP}_stress_tests"
mkdir -p "$OUT_DIR"

CMD=("python" "$ROOT_DIR/qif_v2_cuda.py" "--data-root" "$DATA_ROOT" "--device" "$DEVICE" "--max-seconds" "$MAX_SECONDS" "--max-bins" "$MAX_BINS" "--fit" "--max-iter" "$MAX_ITER" "--n-starts" "$N_STARTS" "--stress-rank2" "--calib-variants")
if [[ "$FIT_PHI" == "1" ]]; then
  CMD+=("--fit-phi")
fi

{
  echo "# Stress tests (rank2 + calib variants)"
  echo "data_root=$DATA_ROOT"
  echo "device=$DEVICE"
  echo "max_seconds=$MAX_SECONDS"
  echo "max_bins=$MAX_BINS"
  echo "max_iter=$MAX_ITER"
  echo "n_starts=$N_STARTS"
  echo "fit_phi=$FIT_PHI"
  echo
} > "$OUT_DIR/run_meta.txt"

set -x
"${CMD[@]}" | tee "$OUT_DIR/stress_tests.log"
set +x

echo "Saved logs under: $OUT_DIR"
