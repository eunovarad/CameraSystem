#!/usr/bin/env python3
# coordinate_anchors.py
#
# Build a camera-derived phantom coordinate table from previously saved click files.
#
# What this script does:
#   1. Loads saved click files from manual_extrinsics.py (clicks_cam0/1/2.npz)
#   2. Loads camera intrinsics and the solved rig (rig_from_clicks.npz/json)
#   3. Triangulates every common fiducial ID seen by all 3 cameras
#   4. Defines an independent coordinate frame using:
#        - Point 1  as the origin
#        - Point 18 to define the X axis direction (negative X like your legacy table)
#        - Point 19 to define the Y axis direction
#      and orthonormalizes the frame with Gram-Schmidt
#   5. Writes an Excel spreadsheet in the format:
#        Step | X Location (mm) | Y Location (mm) | Z Location (mm)
#
# Notes:
#   - This script does NOT use the CMM phantom coordinates as ground truth.
#   - It only uses camera geometry + your previously saved clicks.
#   - It preserves the traditional sign convention so point 18 tends toward negative X
#     and point 19 tends toward positive Y.
#   - If the elevated points (28-47) end up with negative Z, the script flips Z so
#     those points become positive, matching the usual phantom convention.
#
# Typical workflow:
#   1. Run manual_extrinsics.py and save clicks to ./data/clicks/clicks_cam0/1/2.npz
#   2. Run this script
#   3. Compare the exported camera-derived coordinates/distances against CMM yearly files
#
import os
import json
import numpy as np
import pandas as pd
import cv2

# =========================
# PRESET CONFIGURATION
# =========================
CLICK_DIR = "./data/clicks"
CLICK_FILES = [
    os.path.join(CLICK_DIR, "clicks_cam0.npz"),
    os.path.join(CLICK_DIR, "clicks_cam1.npz"),
    os.path.join(CLICK_DIR, "clicks_cam2.npz"),
]

INTRINSIC_FILES = [
    "intrin_cam_left.npz",
    "intrin_cam_right.npz",
    "intrin_cam_back.npz",
]

RIG_FILE = "rig_from_clicks.npz"   # can also be .json

OUTPUT_XLSX = "./data/camera_derived_points_set01.xlsx"
OUTPUT_CSV = "./data/camera_derived_points_set01.csv"

# Coordinate frame anchors
ORIGIN_ID = 1
X_AXIS_ID = 18   # point 18 should land on the negative X side
Y_AXIS_ID = 19   # point 19 should land on the positive Y side

ROUND_DECIMALS = 2
ZERO_TOL_MM = 0.005

# Optional: include a second sheet with pairwise distances for independent CMM comparison
WRITE_DISTANCE_SHEET = True
DISTANCE_SHEET_NAME = "Pairwise Distances"
POINTS_SHEET_NAME = "Camera Coordinates"


# =========================
# IO HELPERS
# =========================
def load_intrinsics_npz(path):
    d = np.load(path, allow_pickle=True)

    if "camera_matrix" in d:
        K = d["camera_matrix"]
    elif "K" in d:
        K = d["K"]
    else:
        raise RuntimeError(f"{path} is missing camera matrix (expected 'camera_matrix' or 'K')")

    if "dist_coeffs" in d:
        D = d["dist_coeffs"]
    elif "D" in d:
        D = d["D"]
    else:
        raise RuntimeError(f"{path} is missing distortion coefficients (expected 'dist_coeffs' or 'D')")

    if "image_size" in d:
        image_size = tuple(np.array(d["image_size"]).reshape(-1)[:2])
    else:
        raise RuntimeError(f"{path} is missing image_size")

    return np.asarray(K, float), np.asarray(D, float), tuple(int(v) for v in image_size)


def load_rig(path):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        H01 = np.asarray(d["H_cam1_in_cam0"], float)
        H02 = np.asarray(d["H_cam2_in_cam0"], float)
        return H01, H02
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        H01 = np.asarray(j["H_cam1_in_cam0"], float)
        H02 = np.asarray(j["H_cam2_in_cam0"], float)
        return H01, H02
    else:
        raise ValueError("Rig file must be .npz or .json")


def load_click_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing click file: {path}")
    d = np.load(path, allow_pickle=True)
    if "clicks" not in d or "ids" not in d:
        raise RuntimeError(f"{path} must contain arrays: clicks, ids")
    clicks = np.asarray(d["clicks"], float)
    ids = np.asarray(d["ids"], int)
    if clicks.shape[0] != ids.shape[0]:
        raise RuntimeError(f"{path}: clicks and ids length mismatch")
    return {int(fid): clicks[i] for i, fid in enumerate(ids)}


# =========================
# GEOMETRY HELPERS
# =========================
def ray_from_pixel(uv, K, D):
    pts = np.array(uv, dtype=np.float64).reshape(1, 1, 2)
    norm = cv2.undistortPoints(pts, K, D)
    x, y = norm.reshape(2)
    d = np.array([x, y, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)
    return d


def triangulate_least_squares(C_list, d_list):
    I = np.eye(3)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for C, d in zip(C_list, d_list):
        P = I - np.outer(d, d)
        A += P
        b += P @ C
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X, *_ = np.linalg.lstsq(A, b, rcond=None)
    return X


def normalize(v, name="vector"):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise RuntimeError(f"Cannot normalize {name}; norm is ~0")
    return v / n


def zero_small_values(arr, tol=ZERO_TOL_MM):
    arr = np.asarray(arr, float).copy()
    arr[np.abs(arr) < tol] = 0.0
    return arr


# =========================
# MAIN PROCESS
# =========================
def main():
    os.makedirs(os.path.dirname(OUTPUT_XLSX) or ".", exist_ok=True)

    # Load inputs
    Ks, Ds, sizes = zip(*[load_intrinsics_npz(p) for p in INTRINSIC_FILES])
    H01, H02 = load_rig(RIG_FILE)
    clicks_by_cam = [load_click_file(p) for p in CLICK_FILES]

    # Determine common IDs seen by all 3 cameras
    common_ids = sorted(set(clicks_by_cam[0].keys()) & set(clicks_by_cam[1].keys()) & set(clicks_by_cam[2].keys()))
    if not common_ids:
        raise RuntimeError("No common IDs found across all 3 click files")

    # Ensure required frame-defining points exist
    for req in [ORIGIN_ID, X_AXIS_ID, Y_AXIS_ID]:
        if req not in common_ids:
            raise RuntimeError(f"Required frame-defining point {req} is missing from common clicked IDs")

    # Extract camera transforms in cam0 frame
    R01, t01 = H01[:3, :3], H01[:3, 3]
    R02, t02 = H02[:3, :3], H02[:3, 3]

    C0 = np.zeros(3)
    C1 = -R01.T @ t01
    C2 = -R02.T @ t02

    # Triangulate all points in cam0 frame
    points_cam0 = {}
    for fid in common_ids:
        p0 = clicks_by_cam[0][fid]
        p1 = clicks_by_cam[1][fid]
        p2 = clicks_by_cam[2][fid]

        d0 = ray_from_pixel(p0, Ks[0], Ds[0])
        d1c = ray_from_pixel(p1, Ks[1], Ds[1])
        d2c = ray_from_pixel(p2, Ks[2], Ds[2])

        d1 = R01.T @ d1c
        d1 /= np.linalg.norm(d1)
        d2 = R02.T @ d2c
        d2 /= np.linalg.norm(d2)

        X = triangulate_least_squares([C0, C1, C2], [d0, d1, d2])
        points_cam0[fid] = X

    # Define independent coordinate frame from points 1, 18, 19
    P1 = points_cam0[ORIGIN_ID]
    P18 = points_cam0[X_AXIS_ID]
    P19 = points_cam0[Y_AXIS_ID]

    # X axis chosen so point 18 lands on negative X, matching the legacy table
    x_hat = normalize(P1 - P18, name="x axis (P1 - P18)")

    # Y axis from point 19, orthogonalized against x_hat
    y_temp = P19 - P1
    y_temp = y_temp - np.dot(y_temp, x_hat) * x_hat
    y_hat = normalize(y_temp, name="orthogonalized y axis from P19")

    # Right-handed Z
    z_hat = normalize(np.cross(x_hat, y_hat), name="z axis")

    # Re-orthogonalize Y so the frame is perfectly orthonormal
    y_hat = normalize(np.cross(z_hat, x_hat), name="re-orthogonalized y axis")

    # If elevated points (28-47) end up negative in Z, flip Y/Z together to preserve handedness
    elevated_ids = [fid for fid in common_ids if fid >= 28]
    if elevated_ids:
        z_vals = [np.dot(points_cam0[fid] - P1, z_hat) for fid in elevated_ids]
        if np.mean(z_vals) < 0:
            y_hat *= -1.0
            z_hat *= -1.0

    # Transform all points into the camera-derived phantom frame
    rows = []
    coord_dict = {}
    for fid in common_ids:
        p = points_cam0[fid] - P1
        x = float(np.dot(p, x_hat))
        y = float(np.dot(p, y_hat))
        z = float(np.dot(p, z_hat))
        xyz = zero_small_values(np.array([x, y, z]))
        xyz = np.round(xyz, ROUND_DECIMALS)
        coord_dict[fid] = xyz
        rows.append([fid, xyz[0], xyz[1], xyz[2]])

    rows.sort(key=lambda r: int(r[0]))
    df_points = pd.DataFrame(rows, columns=["Step", "X Location (mm)", "Y Location (mm)", "Z Location (mm)"])

    # Optional pairwise distance sheet for CMM comparison
    if WRITE_DISTANCE_SHEET:
        dist_rows = []
        fids = sorted(coord_dict.keys())
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fi, fj = fids[i], fids[j]
                di = coord_dict[fi]
                dj = coord_dict[fj]
                dist = float(np.linalg.norm(dj - di))
                dist_rows.append([fi, fj, round(dist, 4)])
        df_dists = pd.DataFrame(dist_rows, columns=["Step A", "Step B", "Distance (mm)"])

    # Write outputs
    df_points.to_csv(OUTPUT_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_points.to_excel(writer, index=False, sheet_name=POINTS_SHEET_NAME)
        if WRITE_DISTANCE_SHEET:
            df_dists.to_excel(writer, index=False, sheet_name=DISTANCE_SHEET_NAME)

    # Console summary
    print("\nSaved camera-derived coordinates:")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  XLSX: {OUTPUT_XLSX}")
    print("\nFrame definition:")
    print(f"  Origin  = Step {ORIGIN_ID}")
    print(f"  X axis  = from Step {X_AXIS_ID} toward Step {ORIGIN_ID} (so Step {X_AXIS_ID} is negative X)")
    print(f"  Y axis  = from Step {ORIGIN_ID} toward Step {Y_AXIS_ID} (orthogonalized)")
    print(f"  Z axis  = X × Y (flipped if needed so elevated steps are positive Z)")

    # Print the anchor coordinates as a sanity check
    for fid in [ORIGIN_ID, X_AXIS_ID, Y_AXIS_ID]:
        xyz = coord_dict[fid]
        print(f"  Step {fid:>2}: X={xyz[0]:>8.2f}, Y={xyz[1]:>8.2f}, Z={xyz[2]:>8.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
