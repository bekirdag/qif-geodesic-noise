#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST="${MANIFEST:-$ROOT_DIR/data_manifest.tsv}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data}"
BASE_URL="${BASE_URL:-http://et-origin.cism.ucl.ac.be}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Download in manifest order so datasets are fetched one by one.
while IFS=$'\t' read -r dataset filename; do
  [[ -z "${dataset}" || "${dataset}" =~ ^# ]] && continue
  if [[ -z "${filename}" ]]; then
    echo "Skipping malformed line: dataset='$dataset' filename='$filename'" >&2
    continue
  fi

  url="${BASE_URL%/}/MDC1/v2/loudests/${dataset}/${filename}"
  dest_dir="$OUT_DIR/$dataset"
  dest="$dest_dir/$filename"

  mkdir -p "$dest_dir"

  if [[ -f "$dest" ]]; then
    echo "[skip] $dataset/$filename"
    continue
  fi

  echo "[get] $url"
  # Use resume when possible; fall back to fresh download if resume fails.
  if ! curl -L --fail --retry 3 --retry-delay 2 -C - -o "$dest" "$url"; then
    echo "[retry] fresh download for $dataset/$filename" >&2
    rm -f "$dest"
    curl -L --fail --retry 3 --retry-delay 2 -o "$dest" "$url"
  fi

done < "$MANIFEST"

echo "Done. Data available under: $OUT_DIR"
