#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install -r requirements-plot.txt"
    ) from exc

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _to_float(value: str):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_tables(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [_strip_ansi(line.rstrip("\n")) for line in f]

    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and "DATASET" in line:
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 1
            while i < len(lines):
                current = lines[i]
                if current.startswith("+"):
                    i += 1
                    continue
                if not current.startswith("|"):
                    break
                parts = [p.strip() for p in current.strip().strip("|").split("|")]
                if len(parts) != len(cols):
                    i += 1
                    continue
                if parts[0].upper() == "DATASET":
                    i += 1
                    continue
                row = {cols[idx]: parts[idx] for idx in range(len(cols))}
                row["_source"] = path
                rows.append(row)
                i += 1
            continue
        i += 1
    return rows


def _discover_logs(runs_dir: str) -> list[str]:
    logs = []
    for root, _, files in os.walk(runs_dir):
        for name in files:
            if name.endswith(".log"):
                logs.append(os.path.join(root, name))
    return sorted(logs)


def _normalize_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        item = dict(row)
        if "BINS" in row:
            item["BINS"] = _to_int(row.get("BINS"))
        if "R" in row:
            item["R"] = _to_int(row.get("R"))
        if "LR" in row:
            item["LR"] = _to_float(row.get("LR"))
        if "P" in row:
            item["P"] = _to_float(row.get("P"))
        if "P_SE" in row:
            item["P_SE"] = _to_float(row.get("P_SE"))
        if "LNL_ENV" in row:
            item["LNL_ENV"] = _to_float(row.get("LNL_ENV"))
        if "LNL_QIF" in row:
            item["LNL_QIF"] = _to_float(row.get("LNL_QIF"))
        if "DUR_S" in row:
            item["DUR_S"] = _to_float(row.get("DUR_S"))
        normalized.append(item)
    return normalized


def _plot_lr_vs_bins(rows: list[dict], out_path: str) -> bool:
    bins_to_lr = defaultdict(list)
    for row in rows:
        bins = row.get("BINS")
        lr = row.get("LR")
        if bins is None or lr is None:
            continue
        bins_to_lr[bins].append(lr)

    if not bins_to_lr:
        return False

    bins_sorted = sorted(bins_to_lr)
    lr_vals = [float(np.median(bins_to_lr[b])) for b in bins_sorted]

    plt.figure(figsize=(6, 4))
    plt.plot(bins_sorted, lr_vals, marker="o")
    plt.xlabel("Bins")
    plt.ylabel("LR (median across groups)")
    plt.title("Resolution Sensitivity (LR vs Bins)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def _plot_lr_by_dataset(rows: list[dict], out_path: str) -> bool:
    rows_by_source = defaultdict(list)
    for row in rows:
        dataset = row.get("DATASET")
        if not dataset:
            continue
        rows_by_source[row.get("_source")].append(row)

    if not rows_by_source:
        return False

    best_source = max(rows_by_source.items(), key=lambda kv: len(kv[1]))[0]
    selected = rows_by_source[best_source]

    labels = []
    values = []
    for row in selected:
        lr = row.get("LR")
        if lr is None:
            continue
        dataset = row.get("DATASET", "")
        gps = row.get("GPS", "")
        label = f"{dataset}@{gps}" if gps else dataset
        labels.append(label)
        values.append(lr)

    if not labels:
        return False

    plt.figure(figsize=(9, 4))
    x = np.arange(len(labels))
    plt.bar(x, values, color="#4c72b0")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("LR")
    plt.title("LR by Dataset Group")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def _plot_bootstrap_p(rows: list[dict], out_path: str) -> bool:
    points = []
    labels = []
    for row in rows:
        p = row.get("P")
        if p is None:
            continue
        points.append(p)
        dataset = row.get("DATASET", "")
        gps = row.get("GPS", "")
        labels.append(f"{dataset}@{gps}" if gps else dataset)

    if not points:
        return False

    plt.figure(figsize=(6, 3.5))
    x = np.arange(len(points))
    plt.scatter(x, points, color="#dd8452")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Bootstrap p-value")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Bootstrap p-values")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def _plot_stress_variants(standard_rows: list[dict], variant_rows: list[dict], out_path: str) -> bool:
    variant_map = {}

    for row in variant_rows:
        name = row.get("VARIANT")
        lr = row.get("LR")
        if name and lr is not None:
            variant_map[name] = lr

    if not variant_map:
        return False

    for row in standard_rows:
        source = row.get("_source", "")
        if "stress_tests" in source and row.get("LR") is not None:
            variant_map.setdefault("baseline", row.get("LR"))

    for row in standard_rows:
        source = row.get("_source", "")
        if "mask_transfer" in source and row.get("LR") is not None:
            variant_map.setdefault("line_mask_transfer", row.get("LR"))

    names = list(variant_map.keys())
    values = [variant_map[name] for name in names]

    plt.figure(figsize=(6, 3.5))
    x = np.arange(len(names))
    plt.bar(x, values, color="#55a868")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("LR")
    plt.title("Stress Test Variants")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def _plot_psd(args: argparse.Namespace) -> None:
    gammas = [float(g.strip()) for g in args.gammas.split(",") if g.strip()]
    freqs = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.n)

    plt.figure(figsize=(6, 4))
    for gamma in gammas:
        psd = args.A_h * (freqs / args.f0) ** (-gamma)
        plt.loglog(freqs, psd, label=f"gamma={gamma}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("S_h(f) [arb. units]")
    plt.title("Phenomenological PSD Models")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()


def _plot_injection(args: argparse.Namespace) -> None:
    alphas = []
    lrs = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha = _to_float(row.get("alpha_injected"))
            lr = _to_float(row.get("LR"))
            if alpha is None or lr is None:
                continue
            alphas.append(alpha)
            lrs.append(lr)

    if not alphas:
        raise SystemExit("No usable rows found in the injection CSV.")

    order = np.argsort(alphas)
    alphas = np.array(alphas)[order]
    lrs = np.array(lrs)[order]

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, lrs, marker="o")
    if args.logx and args.symlog:
        raise SystemExit("Use only one of --logx or --symlog.")
    if args.logx:
        if np.any(alphas <= 0):
            keep = alphas > 0
            if not np.any(keep):
                raise SystemExit("No positive alpha values available for --logx.")
            skipped = int(np.sum(~keep))
            print(
                f"Skipping {skipped} non-positive alpha values for --logx.",
                file=sys.stderr,
            )
            alphas = alphas[keep]
            lrs = lrs[keep]
        plt.xscale("log")
    elif args.symlog:
        plt.xscale("symlog", linthresh=args.linthresh)
    plt.xlabel("Injected alpha")
    plt.ylabel("LR")
    plt.title("Injection Recovery Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures for the QIF geodesic-noise paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validation = subparsers.add_parser("validation", help="Plot validation figures from test-runs logs.")
    validation.add_argument("--runs-dir", default="test-runs", help="Directory containing run logs.")
    validation.add_argument("--out-dir", default="figures", help="Output directory for plots.")

    psd = subparsers.add_parser("psd", help="Plot phenomenological PSD curves.")
    psd.add_argument("--A-h", dest="A_h", type=float, default=1.0, help="Amplitude A_h.")
    psd.add_argument("--f0", type=float, default=1.0, help="Reference frequency f0 (Hz).")
    psd.add_argument("--fmin", type=float, default=1e-1, help="Minimum frequency (Hz).")
    psd.add_argument("--fmax", type=float, default=1e3, help="Maximum frequency (Hz).")
    psd.add_argument("--gammas", type=str, default="2,1,3", help="Comma-separated gamma values.")
    psd.add_argument("--n", type=int, default=400, help="Number of frequency points.")
    psd.add_argument("--out", type=str, default="figures/model_psd.png", help="Output path.")

    injection = subparsers.add_parser("injection", help="Plot injection recovery curve from CSV.")
    injection.add_argument("--csv", default="test-runs/injection_sweep.csv", help="CSV file from run_injection_test.py.")
    injection.add_argument("--out", default="figures/injection_curve.png", help="Output path.")
    injection.add_argument("--logx", action="store_true", help="Log-scale x-axis.")
    injection.add_argument("--symlog", action="store_true", help="Symmetric log x-axis (keeps zero).")
    injection.add_argument(
        "--linthresh",
        type=float,
        default=1e-2,
        help="Symlog linear threshold (used with --symlog).",
    )

    args = parser.parse_args()

    if args.command == "validation":
        os.makedirs(args.out_dir, exist_ok=True)
        logs = _discover_logs(args.runs_dir)
        all_rows = []
        for path in logs:
            all_rows.extend(_parse_tables(path))
        rows = _normalize_rows(all_rows)

        standard_rows = [row for row in rows if "VARIANT" not in row]
        variant_rows = [row for row in rows if "VARIANT" in row]

        outputs = []
        outputs.append(
            _plot_lr_vs_bins(standard_rows, os.path.join(args.out_dir, "lr_vs_bins.png"))
        )
        outputs.append(
            _plot_lr_by_dataset(standard_rows, os.path.join(args.out_dir, "lr_by_dataset.png"))
        )
        outputs.append(
            _plot_bootstrap_p(standard_rows, os.path.join(args.out_dir, "bootstrap_p_values.png"))
        )
        outputs.append(
            _plot_stress_variants(
                standard_rows, variant_rows, os.path.join(args.out_dir, "stress_variants.png")
            )
        )

        if not any(outputs):
            raise SystemExit("No plots were generated; check that logs exist in the runs directory.")

    elif args.command == "psd":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        _plot_psd(args)

    elif args.command == "injection":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        _plot_injection(args)


if __name__ == "__main__":
    main()
