#!/usr/bin/env python3
# coordinate_plane_fit.py
#
# Build a camera-derived phantom coordinate table from previously saved click files,
# using a best-fit Z=0 plane through steps 1–27.
#
# Coordinate frame definition:
#   - Origin  : Step 1
#   - Z=0 plane: best-fit plane through all available points among steps 1–27
#   - X axis  : projection of (Step 18 - Step 1) into that plane,
#               oriented so Step 18 lands on negative X (matching legacy convention)
#   - Y axis  : projection of (Step 19 - Step 1) into that plane,
#               orthogonalized within the plane and oriented so Step 19 lands on positive Y
#   - Z axis  : plane normal, flipped if needed so elevated steps (28–47) are positive Z
#
# Outputs:
#   1. Excel workbook with a sheet in the format:
#        Step | X Location (mm) | Y Location (mm) | Z Location (mm)
#      matching the style of the CMM export
#   2. CSV with the same table
#   3. Optional pairwise distance sheet for independent CMM comparison
#   4. Optional frame metadata sheet for sanity checking the constructed coordinate frame
#
# Requirements:
#   - Saved clicks from manual_extrinsics.py in ./data/clicks/clicks_cam0.npz etc.
#   - Intrinsics npz files (left/right/back)
#   - rig_from_clicks.npz or .json
#   - pandas + openpyxl installed for Excel output
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

OUTPUT_XLSX = "./data/camera_derived_points_set01_planefit.xlsx"
OUTPUT_CSV = "./data/camera_derived_points_set01_planefit.csv"

# Frame anchors
ORIGIN_ID = 1
X_AXIS_ID = 18
Y_AXIS_ID = 19
PLANE_Z0_IDS = list(range(1, 28))  # steps 1–27 should lie on nominal Z=0

# Output formatting
ROUND_DECIMALS = 2
ZERO_TOL_MM = 0.005
WRITE_DISTANCE_SHEET = True
WRITE_FRAME_INFO_SHEET = True
POINTS_SHEET_NAME = "Camera Coordinates"
DISTANCE_SHEET_NAME = "Pairwise Distances"
FRAMEINFO_SHEET_NAME = "Frame Info"


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
        raise RuntimeError(f"{path} missing camera matrix (expected 'camera_matrix' or 'K')")

    if "dist_coeffs" in d:
        D = d["dist_coeffs"]
    elif "D" in d:
        D = d["D"]
    else:
        raise RuntimeError(f"{path} missing distortion coefficients (expected 'dist_coeffs' or 'D')")

    if "image_size" in d:
        image_size = tuple(np.array(d["image_size"]).reshape(-1)[:2])
    else:
        raise RuntimeError(f"{path} missing image_size")

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


def fit_plane_svd(points_xyz):
    """
    Fit a plane to Nx3 points using SVD.
    Returns:
      centroid, normal, rms_to_plane
    """
    P = np.asarray(points_xyz, float)
    if P.shape[0] < 3:
        raise RuntimeError("Need at least 3 points to fit a plane")
    centroid = P.mean(axis=0)
    X = P - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1]
    normal = normalize(normal, name="plane normal")
    signed_dist = X @ normal
    rms = float(np.sqrt(np.mean(signed_dist ** 2)))
    return centroid, normal, rms


def project_vector_to_plane(v, n):
    v = np.asarray(v, float)
    n = np.asarray(n, float)
    return v - np.dot(v, n) * n


# =========================
# MAIN PROCESS
# =========================
def main():
    out_dir = os.path.dirname(OUTPUT_XLSX) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    Ks, Ds, _ = zip(*[load_intrinsics_npz(p) for p in INTRINSIC_FILES])
    H01, H02 = load_rig(RIG_FILE)
    clicks_by_cam = [load_click_file(p) for p in CLICK_FILES]

    # Common IDs across all 3 click files
    common_ids = sorted(set(clicks_by_cam[0].keys()) & set(clicks_by_cam[1].keys()) & set(clicks_by_cam[2].keys()))
    if not common_ids:
        raise RuntimeError("No common IDs found across all 3 click files")

    # Required frame-defining IDs
    for req in [ORIGIN_ID, X_AXIS_ID, Y_AXIS_ID]:
        if req not in common_ids:
            raise RuntimeError(f"Required frame-defining point {req} is missing from common clicked IDs")

    # Ensure enough Z=0 plane IDs are present
    plane_ids_present = [fid for fid in PLANE_Z0_IDS if fid in common_ids]
    if len(plane_ids_present) < 3:
        raise RuntimeError("Fewer than 3 of the requested Z=0 plane IDs are present in the click files")

    # Camera transforms into cam0 frame
    R01, t01 = H01[:3, :3], H01[:3, 3]
    R02, t02 = H02[:3, :3], H02[:3, 3]
    C0 = np.zeros(3)
    C1 = -R01.T @ t01
    C2 = -R02.T @ t02

    # Triangulate all points into cam0 frame
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

    # Origin
    P1 = points_cam0[ORIGIN_ID]

    # Best-fit plane through steps 1–27 (or the subset present)
    plane_points = np.vstack([points_cam0[fid] for fid in plane_ids_present])
    plane_centroid, z_hat_raw, plane_rms = fit_plane_svd(plane_points)

    # Orient plane normal so elevated points tend to positive Z
    elevated_ids = [fid for fid in common_ids if fid >= 28]
    if elevated_ids:
        z_scores = [np.dot(points_cam0[fid] - P1, z_hat_raw) for fid in elevated_ids]
        if np.mean(z_scores) < 0:
            z_hat_raw *= -1.0

    z_hat = normalize(z_hat_raw, name="final z axis")

    # X axis from projected P1->P18 direction, then flipped so step 18 is negative X.
    v18 = points_cam0[X_AXIS_ID] - P1
    x_proj = project_vector_to_plane(v18, z_hat)
    x_hat = normalize(-x_proj, name="x axis (-projected P18 direction)")

    # Y axis from projected P1->P19 direction, orthogonalized within plane.
    v19 = points_cam0[Y_AXIS_ID] - P1
    y_proj = project_vector_to_plane(v19, z_hat)
    y_proj = y_proj - np.dot(y_proj, x_hat) * x_hat
    y_hat = normalize(y_proj, name="initial y axis from projected P19 direction")

    # Re-orthogonalize in a clean right-handed way
    z_hat = normalize(np.cross(x_hat, y_hat), name="recomputed z axis")
    y_hat = normalize(np.cross(z_hat, x_hat), name="re-orthogonalized y axis")

    # Final sanity orientation: ensure Step 19 lands on positive Y
    test19 = points_cam0[Y_AXIS_ID] - P1
    if np.dot(test19, y_hat) < 0:
        y_hat *= -1.0
        z_hat *= -1.0

    # Final sanity orientation: ensure elevated points have positive Z on average
    if elevated_ids:
        z_scores = [np.dot(points_cam0[fid] - P1, z_hat) for fid in elevated_ids]
        if np.mean(z_scores) < 0:
            y_hat *= -1.0
            z_hat *= -1.0

    # Transform all common points into the new frame
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

    # Pairwise distance sheet
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

    # Frame info sheet for sanity checks
    if WRITE_FRAME_INFO_SHEET:
        frame_rows = []
        def add_row(k, v):
            frame_rows.append([k, v])
        add_row("Origin ID", ORIGIN_ID)
        add_row("X Axis ID", X_AXIS_ID)
        add_row("Y Axis ID", Y_AXIS_ID)
        add_row("Z=0 plane IDs used", ", ".join(str(fid) for fid in plane_ids_present))
        add_row("Plane fit RMS to Z=0 IDs (mm)", round(plane_rms, 6))
        add_row("Step 18 coordinate", str(tuple(coord_dict[X_AXIS_ID].tolist())))
        add_row("Step 19 coordinate", str(tuple(coord_dict[Y_AXIS_ID].tolist())))
        add_row("x_hat", np.array2string(x_hat, precision=6, separator=', '))
        add_row("y_hat", np.array2string(y_hat, precision=6, separator=', '))
        add_row("z_hat", np.array2string(z_hat, precision=6, separator=', '))
        df_frame = pd.DataFrame(frame_rows, columns=["Field", "Value"])

    # Write outputs
    df_points.to_csv(OUTPUT_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_points.to_excel(writer, index=False, sheet_name=POINTS_SHEET_NAME)
        if WRITE_DISTANCE_SHEET:
            df_dists.to_excel(writer, index=False, sheet_name=DISTANCE_SHEET_NAME)
        if WRITE_FRAME_INFO_SHEET:
            df_frame.to_excel(writer, index=False, sheet_name=FRAMEINFO_SHEET_NAME)

    # Console summary
    print("\nSaved plane-fit camera-derived coordinates:")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  XLSX: {OUTPUT_XLSX}")
    print("\nFrame definition:")
    print(f"  Origin  = Step {ORIGIN_ID}")
    print(f"  Z=0 plane = best-fit through steps {plane_ids_present[0]}–{plane_ids_present[-1]} (subset present: {len(plane_ids_present)} points)")
    print(f"  X axis  = projection of Step {X_AXIS_ID} into the plane, oriented so Step {X_AXIS_ID} is negative X")
    print(f"  Y axis  = projection of Step {Y_AXIS_ID} into the plane, orthogonalized within the plane, oriented so Step {Y_AXIS_ID} is positive Y")
    print(f"  Z axis  = plane normal, flipped if needed so elevated steps are positive Z")
    print(f"  Plane fit RMS (steps 1–27 to Z=0 plane): {plane_rms:.6f} mm")
    print("\nAnchor coordinates in final frame:")
    for fid in [ORIGIN_ID, X_AXIS_ID, Y_AXIS_ID]:
        xyz = coord_dict[fid]
        print(f"  Step {fid:>2}: X={xyz[0]:>8.2f}, Y={xyz[1]:>8.2f}, Z={xyz[2]:>8.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
