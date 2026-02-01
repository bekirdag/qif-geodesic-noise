#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Example orchestration; override env vars as needed.
# DATA_ROOT, DEVICE, MAX_SECONDS, MAX_BINS, BOOTSTRAP_N, etc.

"$ROOT_DIR/scripts/run_sensitivity_sweep.sh"
"$ROOT_DIR/scripts/run_bootstrap.sh"
"$ROOT_DIR/scripts/run_stress_tests.sh"
"$ROOT_DIR/scripts/run_line_mask_transfer.sh"
"$ROOT_DIR/scripts/run_injection_test.py"

