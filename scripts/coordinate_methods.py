#!/usr/bin/env python3
# coordinate_methods.py
#
# Unified camera-derived coordinate generator with three selectable methods:
#
#   1) anchors
#      - Origin = Step 1
#      - X axis from Step 18 toward Step 1 (so Step 18 is negative X)
#      - Y axis from Step 1 toward Step 19, orthogonalized against X
#      - Z axis = X × Y
#
#   2) plane_fit
#      - Origin = Step 1
#      - Z=0 plane = best-fit plane through Steps 1–27
#      - X axis = projection of Step 18 direction into that plane, oriented so Step 18 is negative X
#      - Y axis = projection of Step 19 direction into that plane, orthogonalized in the plane, oriented so Step 19 is positive Y
#      - Z axis = plane normal
#
#   3) z_level_affine
#      - Same frame construction as plane_fit for X/Y/Z orientation
#      - Then applies a global affine correction to Z using trusted nominal Z levels:
#            0, 10, 15, 36, 43, 50, 56 mm
#        based on the average measured Z of each level group.
#      - This preserves X/Y and keeps Z monotonic/continuous, while using all trusted levels
#        to reduce vertical bias/tilt in a principled way.
#
# Outputs:
#   - Excel workbook with the main sheet formatted as:
#         Step | X Location (mm) | Y Location (mm) | Z Location (mm)
#   - CSV with the same table
#   - Optional sheets: pairwise distances, frame info, z-level fit summary
#
# Notes:
#   - This script does NOT use CMM XYZ point coordinates as ground truth.
#   - It only uses camera triangulation plus your chosen frame-building method.
#   - Method 3 uses only the trusted nominal Z levels as a 1D calibration along Z.
#
# Typical workflow:
#   1) Run manual_extrinsics.py and save clicks to ./data/clicks/clicks_cam0/1/2.npz
#   2) Choose METHOD below
#   3) Run this script
#   4) Compare outputs across methods/trials/CMM candidates
#
import os
import json
import numpy as np
import pandas as pd
import cv2

# =========================
# PRESET CONFIGURATION
# =========================
METHOD = "plane_fit"  # choose: "anchors", "plane_fit", "z_level_affine"
SET_TAG = "set01"

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

OUTPUT_XLSX = f"./data/camera_derived_points_{SET_TAG}_{METHOD}.xlsx"
OUTPUT_CSV = f"./data/camera_derived_points_{SET_TAG}_{METHOD}.csv"

# Coordinate frame anchors
ORIGIN_ID = 1
X_AXIS_ID = 18   # point 18 should land on the negative X side
Y_AXIS_ID = 19   # point 19 should land on the positive Y side
PLANE_Z0_IDS = list(range(1, 28))  # steps 1–27 should lie on nominal Z=0

# Trusted nominal Z levels for method 3
Z_LEVEL_GROUPS = {
    0.0:  list(range(1, 28)),        # steps 1–27
    10.0: [28, 29, 30],
    15.0: [31, 32, 33, 34],
    36.0: [35, 36, 37],
    43.0: [38, 39],
    50.0: [40, 41, 42],
    56.0: [43, 44, 45, 46, 47],
}

ROUND_DECIMALS = 2
ZERO_TOL_MM = 0.005
WRITE_DISTANCE_SHEET = True
WRITE_FRAME_INFO_SHEET = True
WRITE_ZLEVEL_SHEET = True
POINTS_SHEET_NAME = "Camera Coordinates"
DISTANCE_SHEET_NAME = "Pairwise Distances"
FRAMEINFO_SHEET_NAME = "Frame Info"
ZLEVEL_SHEET_NAME = "Z-Level Summary"


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


def fit_plane_svd(points_xyz):
    P = np.asarray(points_xyz, float)
    if P.shape[0] < 3:
        raise RuntimeError("Need at least 3 points to fit a plane")
    centroid = P.mean(axis=0)
    X = P - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = normalize(Vt[-1], name="plane normal")
    signed_dist = X @ normal
    rms = float(np.sqrt(np.mean(signed_dist ** 2)))
    return centroid, normal, rms


def project_vector_to_plane(v, n):
    v = np.asarray(v, float)
    n = np.asarray(n, float)
    return v - np.dot(v, n) * n


def pairwise_distances_table(coord_dict):
    rows = []
    fids = sorted(coord_dict.keys())
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            fi, fj = fids[i], fids[j]
            di = coord_dict[fi]
            dj = coord_dict[fj]
            dist = float(np.linalg.norm(dj - di))
            rows.append([fi, fj, round(dist, 4)])
    return pd.DataFrame(rows, columns=["Step A", "Step B", "Distance (mm)"])


# =========================
# FRAME-BUILDING METHODS
# =========================
def build_frame_anchors(points_cam0):
    P1 = points_cam0[ORIGIN_ID]
    P18 = points_cam0[X_AXIS_ID]
    P19 = points_cam0[Y_AXIS_ID]

    x_hat = normalize(P1 - P18, name="x axis (P1 - P18)")
    y_temp = P19 - P1
    y_temp = y_temp - np.dot(y_temp, x_hat) * x_hat
    y_hat = normalize(y_temp, name="orthogonalized y axis from P19")
    z_hat = normalize(np.cross(x_hat, y_hat), name="z axis")
    y_hat = normalize(np.cross(z_hat, x_hat), name="re-orthogonalized y axis")

    elevated_ids = [fid for fid in points_cam0.keys() if fid >= 28]
    if elevated_ids:
        z_vals = [np.dot(points_cam0[fid] - P1, z_hat) for fid in elevated_ids]
        if np.mean(z_vals) < 0:
            y_hat *= -1.0
            z_hat *= -1.0

    meta = {
        "method": "anchors",
        "origin_id": ORIGIN_ID,
        "x_axis_id": X_AXIS_ID,
        "y_axis_id": Y_AXIS_ID,
        "z0_plane_ids_used": "",
        "plane_fit_rms_mm": "",
    }
    return P1, x_hat, y_hat, z_hat, meta


def build_frame_planefit(points_cam0, common_ids):
    plane_ids_present = [fid for fid in PLANE_Z0_IDS if fid in common_ids]
    if len(plane_ids_present) < 3:
        raise RuntimeError("Fewer than 3 of the requested Z=0 plane IDs are present in the click files")

    P1 = points_cam0[ORIGIN_ID]
    plane_points = np.vstack([points_cam0[fid] for fid in plane_ids_present])
    _, z_hat_raw, plane_rms = fit_plane_svd(plane_points)

    elevated_ids = [fid for fid in common_ids if fid >= 28]
    if elevated_ids:
        z_scores = [np.dot(points_cam0[fid] - P1, z_hat_raw) for fid in elevated_ids]
        if np.mean(z_scores) < 0:
            z_hat_raw *= -1.0
    z_hat = normalize(z_hat_raw, name="final z axis")

    v18 = points_cam0[X_AXIS_ID] - P1
    x_proj = project_vector_to_plane(v18, z_hat)
    x_hat = normalize(-x_proj, name="x axis (-projected P18 direction)")

    v19 = points_cam0[Y_AXIS_ID] - P1
    y_proj = project_vector_to_plane(v19, z_hat)
    y_proj = y_proj - np.dot(y_proj, x_hat) * x_hat
    y_hat = normalize(y_proj, name="initial y axis from projected P19 direction")

    z_hat = normalize(np.cross(x_hat, y_hat), name="recomputed z axis")
    y_hat = normalize(np.cross(z_hat, x_hat), name="re-orthogonalized y axis")

    test19 = points_cam0[Y_AXIS_ID] - P1
    if np.dot(test19, y_hat) < 0:
        y_hat *= -1.0
        z_hat *= -1.0

    if elevated_ids:
        z_scores = [np.dot(points_cam0[fid] - P1, z_hat) for fid in elevated_ids]
        if np.mean(z_scores) < 0:
            y_hat *= -1.0
            z_hat *= -1.0

    meta = {
        "method": "plane_fit",
        "origin_id": ORIGIN_ID,
        "x_axis_id": X_AXIS_ID,
        "y_axis_id": Y_AXIS_ID,
        "z0_plane_ids_used": ", ".join(str(fid) for fid in plane_ids_present),
        "plane_fit_rms_mm": round(plane_rms, 6),
    }
    return P1, x_hat, y_hat, z_hat, meta


def transform_points(points_cam0, origin, x_hat, y_hat, z_hat):
    rows = []
    coord_dict = {}
    for fid in sorted(points_cam0.keys()):
        p = points_cam0[fid] - origin
        x = float(np.dot(p, x_hat))
        y = float(np.dot(p, y_hat))
        z = float(np.dot(p, z_hat))
        xyz = zero_small_values(np.array([x, y, z]))
        coord_dict[fid] = xyz
        rows.append([fid, xyz[0], xyz[1], xyz[2]])
    return coord_dict, rows


def apply_z_level_affine(coord_dict, common_ids):
    # Build observed mean Z per trusted nominal level group.
    fit_rows = []
    obs_means = []
    nominals = []

    for nominal_z, ids in Z_LEVEL_GROUPS.items():
        ids_present = [fid for fid in ids if fid in common_ids and fid in coord_dict]
        if not ids_present:
            continue
        z_vals = [coord_dict[fid][2] for fid in ids_present]
        obs_mean = float(np.mean(z_vals))
        obs_means.append(obs_mean)
        nominals.append(float(nominal_z))
        fit_rows.append([nominal_z, ", ".join(str(fid) for fid in ids_present), len(ids_present), obs_mean, None, None])

    if len(obs_means) < 2:
        raise RuntimeError("Need at least two trusted Z levels to compute affine Z correction")

    # Fit nominal = a * observed + b using least squares.
    A = np.column_stack([obs_means, np.ones(len(obs_means))])
    y = np.array(nominals, dtype=float)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    corrected = {}
    for fid, xyz in coord_dict.items():
        new_xyz = np.array(xyz, dtype=float)
        new_xyz[2] = a * new_xyz[2] + b
        corrected[fid] = new_xyz

    corrected = {fid: zero_small_values(xyz) for fid, xyz in corrected.items()}

    # Fill level summary table with corrected means and residuals.
    fit_rows_final = []
    for nominal_z, ids_text, n_used, obs_mean, _, _ in fit_rows:
        ids_present = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
        corr_mean = float(np.mean([corrected[fid][2] for fid in ids_present]))
        residual = corr_mean - nominal_z
        fit_rows_final.append([nominal_z, ids_text, n_used, obs_mean, corr_mean, residual])

    df_zfit = pd.DataFrame(
        fit_rows_final,
        columns=[
            "Trusted Z Level (mm)",
            "Steps Used",
            "N Steps",
            "Observed Mean Z Before Fit (mm)",
            "Mean Z After Fit (mm)",
            "Residual After Fit (mm)",
        ],
    )

    zmeta = {
        "z_affine_slope_a": float(a),
        "z_affine_intercept_b": float(b),
    }
    return corrected, df_zfit, zmeta


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

    # Build frame according to chosen method
    method_key = METHOD.strip().lower()
    if method_key == "anchors":
        origin, x_hat, y_hat, z_hat, meta = build_frame_anchors(points_cam0)
        coord_dict, rows = transform_points(points_cam0, origin, x_hat, y_hat, z_hat)
        df_zfit = None
    elif method_key == "plane_fit":
        origin, x_hat, y_hat, z_hat, meta = build_frame_planefit(points_cam0, common_ids)
        coord_dict, rows = transform_points(points_cam0, origin, x_hat, y_hat, z_hat)
        df_zfit = None
    elif method_key == "z_level_affine":
        origin, x_hat, y_hat, z_hat, meta = build_frame_planefit(points_cam0, common_ids)
        coord_dict_base, rows = transform_points(points_cam0, origin, x_hat, y_hat, z_hat)
        coord_dict, df_zfit, zmeta = apply_z_level_affine(coord_dict_base, common_ids)
        for k, v in zmeta.items():
            meta[k] = v
        rows = [[fid, coord_dict[fid][0], coord_dict[fid][1], coord_dict[fid][2]] for fid in sorted(coord_dict.keys())]
        meta["method"] = "z_level_affine"
    else:
        raise RuntimeError(f"Unknown METHOD '{METHOD}'. Choose anchors, plane_fit, or z_level_affine")

    # Round all outputs late, after any z correction
    rounded_rows = []
    for fid, x, y, z in rows:
        xyz = np.round(zero_small_values(np.array([x, y, z])), ROUND_DECIMALS)
        coord_dict[fid] = xyz
        rounded_rows.append([fid, xyz[0], xyz[1], xyz[2]])

    df_points = pd.DataFrame(rounded_rows, columns=["Step", "X Location (mm)", "Y Location (mm)", "Z Location (mm)"])
    df_points.sort_values("Step", inplace=True)

    # Optional pairwise distance sheet
    df_dists = pairwise_distances_table(coord_dict) if WRITE_DISTANCE_SHEET else None

    # Optional frame info sheet
    df_frame = None
    if WRITE_FRAME_INFO_SHEET:
        frame_rows = []
        def add_row(k, v):
            frame_rows.append([k, v])
        add_row("Method", meta.get("method", method_key))
        add_row("Origin ID", ORIGIN_ID)
        add_row("X Axis ID", X_AXIS_ID)
        add_row("Y Axis ID", Y_AXIS_ID)
        add_row("Z=0 plane IDs used", meta.get("z0_plane_ids_used", ""))
        add_row("Plane fit RMS to Z=0 IDs (mm)", meta.get("plane_fit_rms_mm", ""))
        add_row("Z affine slope a", meta.get("z_affine_slope_a", ""))
        add_row("Z affine intercept b", meta.get("z_affine_intercept_b", ""))
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
        if WRITE_DISTANCE_SHEET and df_dists is not None:
            df_dists.to_excel(writer, index=False, sheet_name=DISTANCE_SHEET_NAME)
        if WRITE_FRAME_INFO_SHEET and df_frame is not None:
            df_frame.to_excel(writer, index=False, sheet_name=FRAMEINFO_SHEET_NAME)
        if method_key == "z_level_affine" and WRITE_ZLEVEL_SHEET and df_zfit is not None:
            df_zfit.to_excel(writer, index=False, sheet_name=ZLEVEL_SHEET_NAME)

    # Console summary
    print("\nSaved camera-derived coordinates:")
    print(f"  Method: {method_key}")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  XLSX: {OUTPUT_XLSX}")
    print("\nFrame definition:")
    if method_key == "anchors":
        print(f"  Origin  = Step {ORIGIN_ID}")
        print(f"  X axis  = from Step {X_AXIS_ID} toward Step {ORIGIN_ID} (so Step {X_AXIS_ID} is negative X)")
        print(f"  Y axis  = from Step {ORIGIN_ID} toward Step {Y_AXIS_ID} (orthogonalized)")
        print(f"  Z axis  = X × Y (flipped if needed so elevated steps are positive Z)")
    else:
        print(f"  Origin  = Step {ORIGIN_ID}")
        print(f"  Z=0 plane = best-fit through steps 1–27 (subset present: {len([fid for fid in PLANE_Z0_IDS if fid in common_ids])} points)")
        print(f"  X axis  = projection of Step {X_AXIS_ID} into the plane, oriented so Step {X_AXIS_ID} is negative X")
        print(f"  Y axis  = projection of Step {Y_AXIS_ID} into the plane, orthogonalized in the plane, oriented so Step {Y_AXIS_ID} is positive Y")
        print(f"  Z axis  = plane normal, flipped if needed so elevated steps are positive Z")
        if meta.get("plane_fit_rms_mm", "") != "":
            print(f"  Plane fit RMS (steps 1–27 to Z=0 plane): {meta['plane_fit_rms_mm']}")
        if method_key == "z_level_affine":
            print(f"  Z affine correction: nominal_z = a * measured_z + b")
            print(f"    a = {meta.get('z_affine_slope_a', '')}")
            print(f"    b = {meta.get('z_affine_intercept_b', '')}")

    print("\nAnchor coordinates in final frame:")
    for fid in [ORIGIN_ID, X_AXIS_ID, Y_AXIS_ID]:
        xyz = coord_dict[fid]
        print(f"  Step {fid:>2}: X={xyz[0]:>8.2f}, Y={xyz[1]:>8.2f}, Z={xyz[2]:>8.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
