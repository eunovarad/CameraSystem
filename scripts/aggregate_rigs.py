#!/usr/bin/env python3
# agregate_rig_estimates.py  (cam0=cam_rt, cam1=cam_back, cam2=cam_lt)
#
# Usage:
#   python agregate_rig_estimates.py rig1.json rig2.json rig3.json ... \
#     --json_out rig_final.json --npz_out rig_final.npz \
#     --intrin_rt intrin_rt_fixed.npz \
#     --intrin_back intrin_back_fixed.npz \
#     --intrin_lt intrin_lt_fixed.npz \
#     --max_cam_rms 5.0
#
# Notes:
# - Drops any input rig whose reprojection_rms_px for ANY cam exceeds --max_cam_rms.
# - Cam mapping is fixed: cam0=cam_rt, cam1=cam_back, cam2=cam_lt.
# - Averages rotations (weighted quaternion mean) and uses median translation.

import argparse, json, numpy as np, sys, os

CAM_LABELS = {0: "cam_rt", 1: "cam_back", 2: "cam_lt"}  # fixed mapping

def load_rig_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # Expected keys: H_cam1_in_cam0, H_cam2_in_cam0, optional reprojection_rms_px
    H01 = np.asarray(d["H_cam1_in_cam0"], dtype=float)
    H02 = np.asarray(d["H_cam2_in_cam0"], dtype=float)
    rms = d.get("reprojection_rms_px", {})
    return H01, H02, rms

def rot_to_quat(R):
    t = np.trace(R)
    w = np.sqrt(max(0.0, 1.0 + t)) / 2.0
    if w < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    x = (R[2,1] - R[1,2]) / (4*w)
    y = (R[0,2] - R[2,0]) / (4*w)
    z = (R[1,0] - R[0,1]) / (4*w)
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def quat_to_rot(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def weighted_quat_mean(Rs, weights):
    Q = np.stack([rot_to_quat(R) for R in Rs], axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    q = (w * Q).sum(axis=0)
    if np.linalg.norm(q) < 1e-12:
        q = Q.mean(axis=0)
    return quat_to_rot(q)

def se3_errors(H_list, H_avg):
    angs, dists = [], []
    for H in H_list:
        dH = np.linalg.inv(H_avg) @ H
        R = dH[:3, :3]; t = dH[:3, 3]
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2.0, -1.0, 1.0)))
        angs.append(ang); dists.append(np.linalg.norm(t))
    return np.array(angs), np.array(dists)

def main():
    ap = argparse.ArgumentParser(description="Aggregate multiple rig JSONs into a robust final rig.")
    ap.add_argument("json_files", nargs="+", help="rig*.json files (cam0=cam_rt, cam1=cam_back, cam2=cam_lt)")
    ap.add_argument("--json_out", default="rig_final.json")
    ap.add_argument("--npz_out", default="rig_final.npz")
    ap.add_argument("--intrin_rt", default="intrin_rt_fixed.npz")
    ap.add_argument("--intrin_back", default="intrin_back_fixed.npz")
    ap.add_argument("--intrin_lt", default="intrin_lt_fixed.npz")
    ap.add_argument("--max_cam_rms", type=float, default=5.0,
                    help="Reject a rig if ANY cam RMS exceeds this (default 5 px)")
    ap.add_argument("--min_samples", type=int, default=2, help="Require at least this many good rigs")
    args = ap.parse_args()

    H01_all, H02_all, keep_idx, drop_idx = [], [], [], []
    weights01, weights02 = [], []

    print(f"[INFO] Cam labels: cam0={CAM_LABELS[0]}, cam1={CAM_LABELS[1]}, cam2={CAM_LABELS[2]}")
    print(f"[INFO] Threshold per-cam RMS <= {args.max_cam_rms:.2f} px")

    for i, p in enumerate(args.json_files):
        try:
            H01, H02, rms = load_rig_json(p)
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
            drop_idx.append((i, p, "read_error"))
            continue

        # Compute a reliability weight from reprojection RMS (lower is better).
        # If RMS dict missing, use neutral weight = 1.
        if isinstance(rms, dict) and len(rms) > 0:
            # Enforce threshold: drop if any cam over limit
            per_cam = {
                0: float(rms.get("cam0", np.nan)),
                1: float(rms.get("cam1", np.nan)),
                2: float(rms.get("cam2", np.nan)),
            }
            bad = any((not np.isfinite(v)) or (v > args.max_cam_rms) for v in per_cam.values())
            if bad:
                drop_idx.append((i, p, f"RMS_over_limit {per_cam}"))
                continue
            # weight = 1 / (eps + mean RMS)
            m = np.mean([per_cam[0], per_cam[1], per_cam[2]])
            w = 1.0 / max(1e-6, m)
        else:
            per_cam = {}
            w = 1.0  # neutral

        H01_all.append(H01)
        H02_all.append(H02)
        weights01.append(w)
        weights02.append(w)
        keep_idx.append((i, p, per_cam if per_cam else "no_rms"))

    if len(H01_all) < args.min_samples:
        print(f"[FATAL] Only {len(H01_all)} good rigs (need >= {args.min_samples}). Kept: {keep_idx}  Dropped: {drop_idx}")
        return 2

    # Stack and extract R,t
    H01_all = np.stack(H01_all, axis=0)
    H02_all = np.stack(H02_all, axis=0)
    R01s = H01_all[:, :3, :3]; t01s = H01_all[:, :3, 3]
    R02s = H02_all[:, :3, :3]; t02s = H02_all[:, :3, 3]

    # Weighted rotation mean, robust (median) translation
    R01 = weighted_quat_mean(R01s, weights01)
    R02 = weighted_quat_mean(R02s, weights02)
    t01 = np.median(t01s, axis=0)
    t02 = np.median(t02s, axis=0)

    H01 = np.eye(4); H01[:3, :3] = R01; H01[:3, 3] = t01
    H02 = np.eye(4); H02[:3, :3] = R02; H02[:3, 3] = t02

    a01, d01 = se3_errors(H01_all, H01)
    a02, d02 = se3_errors(H02_all, H02)

    print(f"[KEEP] {len(keep_idx)} rigs:")
    for _, p, info in keep_idx:
        print(f"       - {os.path.basename(p)}  {info}")
    if drop_idx:
        print(f"[DROP] {len(drop_idx)} rigs:")
        for _, p, why in drop_idx:
            print(f"       - {os.path.basename(p)}  {why}")

    print(f"[SPREAD] (cam1_in_cam0) rot mean={a01.mean():.4f}°  std={a01.std():.4f}° ; trans MAD≈{np.median(np.abs(t01s - t01), axis=0)}")
    print(f"[SPREAD] (cam2_in_cam0) rot mean={a02.mean():.4f}°  std={a02.std():.4f}° ; trans MAD≈{np.median(np.abs(t02s - t02), axis=0)}")

    # Save NPZ
    np.savez(args.npz_out,
             H_cam0=np.eye(4),
             H_cam1_in_cam0=H01, H_cam2_in_cam0=H02,
             H01_samples=H01_all, H02_samples=H02_all,
             rot_err01_deg=a01, rot_err02_deg=a02,
             trans_err01=t01s - t01, trans_err02=t02s - t02,
             kept=np.array([p for _, p, _ in keep_idx]),
             dropped=np.array([p for _, p, _ in drop_idx]))
    print(f"[OK] Wrote {args.npz_out}")

    # Save JSON (compact, with intrinsics + frames + labels)
    def tolist(M): return np.asarray(M, float).tolist()
    out_json = {
        "frames": {"reference": "cam0"},
        "cam_labels": {"cam0": CAM_LABELS[0], "cam1": CAM_LABELS[1], "cam2": CAM_LABELS[2]},
        "intrinsics": {
            "cam0": args.intrin_rt,
            "cam1": args.intrin_back,
            "cam2": args.intrin_lt
        },
        "H_cam1_in_cam0": tolist(H01),
        "H_cam2_in_cam0": tolist(H02),
        "spread_deg": {
            "cam1_in_cam0": {"mean": float(a01.mean()), "std": float(a01.std())},
            "cam2_in_cam0": {"mean": float(a02.mean()), "std": float(a02.std())}
        },
        "num_samples": {"H01": int(H01_all.shape[0]), "H02": int(H02_all.shape[0])}
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"[OK] Wrote {args.json_out}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
