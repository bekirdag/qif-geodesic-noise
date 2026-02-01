#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
DEVICE="${DEVICE:-auto}"
MAX_SECONDS="${MAX_SECONDS:-2048}"
BINS_LIST="${BINS_LIST:-128 256 512}"
MAX_ITER="${MAX_ITER:-120}"
N_STARTS="${N_STARTS:-2}"
FIT_PHI="${FIT_PHI:-1}"

STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="$ROOT_DIR/test-runs/${STAMP}_sensitivity_sweep"
mkdir -p "$OUT_DIR"

CMD_BASE=("python" "$ROOT_DIR/qif_v2_cuda.py" "--data-root" "$DATA_ROOT" "--device" "$DEVICE" "--max-seconds" "$MAX_SECONDS" "--fit" "--max-iter" "$MAX_ITER" "--n-starts" "$N_STARTS")
if [[ "$FIT_PHI" == "1" ]]; then
  CMD_BASE+=("--fit-phi")
fi

{
  echo "# Sensitivity sweep"
  echo "data_root=$DATA_ROOT"
  echo "device=$DEVICE"
  echo "max_seconds=$MAX_SECONDS"
  echo "max_iter=$MAX_ITER"
  echo "n_starts=$N_STARTS"
  echo "fit_phi=$FIT_PHI"
  echo "bins_list=$BINS_LIST"
  echo
} > "$OUT_DIR/run_meta.txt"

for bins in $BINS_LIST; do
  echo "==> max_bins=$bins"
  OUT_FILE="$OUT_DIR/bins_${bins}.log"
  set -x
  "${CMD_BASE[@]}" --max-bins "$bins" | tee "$OUT_FILE"
  set +x
done

echo "Saved logs under: $OUT_DIR"
