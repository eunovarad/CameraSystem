#!/usr/bin/env python3
# make_validation_report.py
#
# Generate a phantom-based accuracy report after you update the rig.
# Inputs:
#   --accuracy   phantom_accuracy.xlsx   (measured vs reference; flexible columns)
#   --rig        rig_final.npz           (optional; included in report metadata only)
#   --aggregate  aggregate_log.txt       (optional; paste stdout from agregate_rig_estimates.py)
#   --out        report_out_dir          (optional; default: report_<timestamp>)
#
# Outputs in OUT_DIR:
#   - summary.csv                     (key stats table)
#   - per_fiducial.csv                (row-by-row deltas & reprojection RMS if present)
#   - hist_dX.png / hist_dY.png / hist_dZ.png / hist_mag.png
#   - scatter_meas_vs_ref_X.png
#   - residuals_3d_quiver.png / residuals_XY.png / residuals_XZ.png / residuals_YZ.png
#   - rig_aggregate.txt               (copied in if provided)
#   - report.md                       (Markdown summary with links to artifacts)

import argparse
import os
import sys
import math
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def norm_cols(df):
    orig = list(df.columns)
    norm = [c.strip().lower().replace(" ", "").replace("_", "") for c in orig]
    return orig, norm

def find_col(df, norms, expect, required=True):
    """
    Find the first column whose normalized name contains all substrings in `expect`.
    """
    orig = list(df.columns)
    for i, c in enumerate(norms):
        if all(s in c for s in expect):
            return orig[i]
    if required:
        raise KeyError(f"Could not find column with tokens {expect} in {list(df.columns)}")
    return None

def median_absolute_deviation(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def safe_numeric(a):
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, s: str):
    path.write_text(s, encoding="utf-8")

# ---------- plotting rules ----------
# Note: 1 chart per figure, default matplotlib colors (per your environment constraints)
def plot_hist(data, bins, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, out_path):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_quiver_3d(Xr, Yr, Zr, dX, dY, dZ, title, out_path, exaggeration=1.0):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xr, Yr, Zr, s=10)
    ax.quiver(Xr, Yr, Zr, dX*exaggeration, dY*exaggeration, dZ*exaggeration, length=1.0, normalize=False)
    ax.set_xlabel("Ref X (mm)"); ax.set_ylabel("Ref Y (mm)"); ax.set_zlabel("Ref Z (mm)")
    ax.set_title(title + (f" (×{exaggeration:g})" if exaggeration!=1.0 else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_residual_projection(Xr, Yr, Zr, dX, dY, title, xlabel, ylabel, out_path, scale=10.0):
    # plot reference plane scatter and 2D vectors (scaled)
    plt.figure()
    plt.scatter(Xr, Yr, s=10)
    plt.quiver(Xr, Yr, dX*scale, dY*scale, angles='xy', scale_units='xy', scale=1.0)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title + f" (vectors ×{scale:g})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Generate phantom validation report.")
    ap.add_argument("--accuracy", required=True, help="phantom_accuracy.xlsx (measured vs reference)")
    ap.add_argument("--rig", default=None, help="rig_final.npz (optional, included in metadata)")
    ap.add_argument("--aggregate", default=None, help="aggregate rig log text (optional)")
    ap.add_argument("--out", default=None, help="output directory (default: report_<timestamp>)")
    ap.add_argument("--quiver_scale", type=float, default=10.0, help="scale factor for 3D residual arrows")
    args = ap.parse_args()

    acc_path = Path(args.accuracy)
    if not acc_path.exists():
        print(f"[ERR] accuracy file not found: {acc_path}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(f"report_{ts}")
    ensure_outdir(out_dir)

    # Load accuracy sheet
    df = pd.read_excel(acc_path)
    orig, norms = norm_cols(df)

    # Column detection (flexible)
    # Prefer phantom-frame explicit names if present
    col_Xm = next((orig[i] for i,c in enumerate(norms) if "x" in c and "meas" in c and "ph" in c), None)
    col_Ym = next((orig[i] for i,c in enumerate(norms) if "y" in c and "meas" in c and "ph" in c), None)
    col_Zm = next((orig[i] for i,c in enumerate(norms) if "z" in c and "meas" in c and "ph" in c), None)
    if col_Xm is None: col_Xm = find_col(df, norms, ["x","meas"])
    if col_Ym is None: col_Ym = find_col(df, norms, ["y","meas"])
    if col_Zm is None: col_Zm = find_col(df, norms, ["z","meas"])

    col_Xr = next((orig[i] for i,c in enumerate(norms) if "x" in c and "ref" in c and "ph" in c), None)
    col_Yr = next((orig[i] for i,c in enumerate(norms) if "y" in c and "ref" in c and "ph" in c), None)
    col_Zr = next((orig[i] for i,c in enumerate(norms) if "z" in c and "ref" in c and "ph" in c), None)
    if col_Xr is None: col_Xr = find_col(df, norms, ["x","ref"])
    if col_Yr is None: col_Yr = find_col(df, norms, ["y","ref"])
    if col_Zr is None: col_Zr = find_col(df, norms, ["z","ref"])

    col_fid = next((orig[i] for i,c in enumerate(norms) if "fid" in c or "id" in c), None)
    col_rms = next((orig[i] for i,c in enumerate(norms) if "rms" in c and "px" in c), None)
    col_rms0 = next((orig[i] for i,c in enumerate(norms) if "rms" in c and "cam0" in c), None)
    col_rms1 = next((orig[i] for i,c in enumerate(norms) if "rms" in c and "cam1" in c), None)
    col_rms2 = next((orig[i] for i,c in enumerate(norms) if "rms" in c and "cam2" in c), None)

    Xr = safe_numeric(df[col_Xr]); Yr = safe_numeric(df[col_Yr]); Zr = safe_numeric(df[col_Zr])
    Xm = safe_numeric(df[col_Xm]); Ym = safe_numeric(df[col_Ym]); Zm = safe_numeric(df[col_Zm])

    dX = Xm - Xr; dY = Ym - Yr; dZ = Zm - Zr
    err = np.sqrt(dX**2 + dY**2 + dZ**2)

    # Stats
    stats = {
        "N": int(np.sum(~np.isnan(err))),
        "mean_|Δ|_mm": float(np.nanmean(err)),
        "median_|Δ|_mm": float(np.nanmedian(err)),
        "mad_|Δ|_mm": float(median_absolute_deviation(err)),
        "p95_|Δ|_mm": float(np.nanpercentile(err, 95)),
        "max_|Δ|_mm": float(np.nanmax(err)),
        "mean_ΔX_mm": float(np.nanmean(dX)),
        "mean_ΔY_mm": float(np.nanmean(dY)),
        "mean_ΔZ_mm": float(np.nanmean(dZ)),
        "std_ΔX_mm": float(np.nanstd(dX)),
        "std_ΔY_mm": float(np.nanstd(dY)),
        "std_ΔZ_mm": float(np.nanstd(dZ)),
    }

    # ΔZ linear trend vs Z_ref: ΔZ = a*Z_ref + b
    A = np.vstack([Zr, np.ones_like(Zr)]).T
    mask = ~np.isnan(dZ) & ~np.isnan(Zr)
    if np.sum(mask) >= 2:
        a, b = np.linalg.lstsq(A[mask], dZ[mask], rcond=None)[0]
        stats["trend_dZ_vs_Zref_slope_mm_per_mm"] = float(a)
        stats["trend_dZ_vs_Zref_intercept_mm"] = float(b)

    # Save per-fiducial table
    out_rows = []
    for i in range(len(err)):
        row = {
            "index": i,
            "fid": int(df[col_fid].iloc[i]) if col_fid is not None and pd.notna(df[col_fid].iloc[i]) else i+1,
            "X_ref_mm": Xr[i], "Y_ref_mm": Yr[i], "Z_ref_mm": Zr[i],
            "X_meas_mm": Xm[i], "Y_meas_mm": Ym[i], "Z_meas_mm": Zm[i],
            "dX_mm": dX[i], "dY_mm": dY[i], "dZ_mm": dZ[i], "|Δ|_mm": err[i],
        }
        if col_rms is not None:  row["RMS_px"] = float(df[col_rms].iloc[i])
        if col_rms0 is not None: row["RMS_cam0_px"] = float(df[col_rms0].iloc[i])
        if col_rms1 is not None: row["RMS_cam1_px"] = float(df[col_rms1].iloc[i])
        if col_rms2 is not None: row["RMS_cam2_px"] = float(df[col_rms2].iloc[i])
        out_rows.append(row)

    per_fid_df = pd.DataFrame(out_rows)
    per_fid_path = out_dir / "per_fiducial.csv"
    per_fid_df.to_csv(per_fid_path, index=False)

    # Plots
    plot_hist(dX, 20, "ΔX Error Histogram", "ΔX (mm)", "Count", out_dir/"hist_dX.png")
    plot_hist(dY, 20, "ΔY Error Histogram", "ΔY (mm)", "Count", out_dir/"hist_dY.png")
    plot_hist(dZ, 20, "ΔZ Error Histogram", "ΔZ (mm)", "Count", out_dir/"hist_dZ.png")
    plot_hist(err, 20, "|Δ| Error Histogram", "|Δ| (mm)", "Count", out_dir/"hist_mag.png")

    plot_scatter(Xr, Xm, "Measured vs Reference (X)", "Reference X (mm)", "Measured X (mm)", out_dir/"scatter_meas_vs_ref_X.png")

    # Residual vectors (3D + 2D projections)
    # Residual vectors (3D + 2D projections)
    plot_quiver_3d(Xr, Yr, Zr, dX, dY, dZ,
                "Residual Vectors (Measured - Reference)",
                out_dir/"residuals_3d_quiver.png",
                exaggeration=args.quiver_scale)

    plot_residual_projection(Xr, Yr, Zr, dX, dY,
                "Residuals in XY", "Ref X (mm)", "Ref Y (mm)",
                out_dir/"residuals_XY.png", scale=args.quiver_scale)

    plot_residual_projection(Xr, Zr, Yr, dX, dZ,
                "Residuals in XZ", "Ref X (mm)", "Ref Z (mm)",
                out_dir/"residuals_XZ.png", scale=args.quiver_scale)

    plot_residual_projection(Yr, Zr, Xr, dY, dZ,
                "Residuals in YZ", "Ref Y (mm)", "Ref Z (mm)",
                out_dir/"residuals_YZ.png", scale=args.quiver_scale)

    # Write summary CSV
    summary_path = out_dir / "summary.csv"
    pd.DataFrame([stats]).to_csv(summary_path, index=False)

    # Copy/record rig + aggregate info if provided
    rig_info = ""
    if args.rig and Path(args.rig).exists():
        rig_info = f"Rig file: {Path(args.rig).resolve()}\n"
    agg_text = ""
    if args.aggregate and Path(args.aggregate).exists():
        agg_text = Path(args.aggregate).read_text(encoding="utf-8", errors="ignore")
        write_text(out_dir/"rig_aggregate.txt", agg_text)

    # Build Markdown report
    md = []
    md.append(f"# Phantom Validation Report\n")
    md.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    md.append(f"- Source accuracy file: `{acc_path}`\n")
    if rig_info: md.append(f"- {rig_info}")
    if args.aggregate: md.append(f"- Aggregate log: `{args.aggregate}`\n")

    md.append("\n## Summary Statistics\n")
    for k in [
        "N","mean_|Δ|_mm","median_|Δ|_mm","mad_|Δ|_mm",
        "p95_|Δ|_mm","max_|Δ|_mm",
        "mean_ΔX_mm","mean_ΔY_mm","mean_ΔZ_mm",
        "std_ΔX_mm","std_ΔY_mm","std_ΔZ_mm",
        "trend_dZ_vs_Zref_slope_mm_per_mm","trend_dZ_vs_Zref_intercept_mm"
    ]:
        if k in stats:
            md.append(f"- **{k}**: {stats[k]:.6g}")

    md.append("\n## Plots\n")
    for f in ["hist_dX.png","hist_dY.png","hist_dZ.png","hist_mag.png",
              "scatter_meas_vs_ref_X.png",
              "residuals_3d_quiver.png","residuals_XY.png","residuals_XZ.png","residuals_YZ.png"]:
        md.append(f"- {f}")

    if agg_text:
        md.append("\n## Rig Aggregation Notes (verbatim)\n")
        md.append("```\n" + agg_text.strip() + "\n```")

    md.append("\n## Files\n")
    md.append(f"- [summary.csv]({summary_path.name})")
    md.append(f"- [per_fiducial.csv]({per_fid_path.name})")

    write_text(out_dir/"report.md", "\n".join(md))

    print(f"[OK] Report written to: {out_dir.resolve()}")
    print(f"      - summary.csv")
    print(f"      - per_fiducial.csv")
    print(f"      - report.md")
    print(f"      - plots: *.png")

if __name__ == "__main__":
    main()
