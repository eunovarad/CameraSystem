#!/usr/bin/env python3
# phantom_accuracy_test_auto.py
#
# Purpose
# -------
# Update of "phantom_accuracy_test_with_modes_fixed.py" to REMOVE the
# "establish phantom pose first (1, 19, +anchor)" requirement.
#
# How it works now
# ----------------
# - You simply measure fiducials by ID (arbitrary or 1→47 order).
# - As soon as you have >= 3 distinct fiducials clicked, the script
#   computes a rigid best‑fit (Kabsch) transform from PHANTOM→cam0
#   using *all* measured correspondences so far (or a fixed anchor set
#   if you pass --anchors), and immediately reports per‑point errors
#   against phantom_points.xlsx in the PHANTOM frame.
# - Before you have 3 points, it will still triangulate and log, but
#   it cannot compare to phantom coordinates yet (insufficient DOF).
#
# Notes
# -----
# - Nothing about your intrinsics/rig changes.
# - Two-click ROI→zoom picking and per‑camera reprojection RMS remain.
# - CSV is written incrementally; early rows (<3 points) won't have
#   PHANTOM-frame comparisons, later rows will.
#
# Example (FILES mode):
#   python phantom_accuracy_test_auto.py \
#     --images capture0/cam0_XXXX.png capture1/cam1_XXXX.png capture2/cam2_XXXX.png \
#     --intrinsics intrinsics_cam0.npz intrinsics_cam1.npz intrinsics_cam2.npz \
#     --rig rig_final.json \
#     --phantom phantom_points.xlsx \
#     --csv_out phantom_accuracy.csv
#
# Example (LIVE mode):
#   python phantom_accuracy_test_auto.py --live --cams 0 1 2 \
#     --intrinsics intrinsics_cam0.npz intrinsics_cam1.npz intrinsics_cam2.npz \
#     --rig rig_final.npz \
#     --phantom phantom_points.xlsx \
#     --out_root captures --csv_out phantom_accuracy.csv
#
import os, sys, time, json, csv, argparse
import numpy as np
import cv2

# ---------------- IO helpers ----------------
def load_intrinsics(paths):
    Ks, Ds, sizes = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Ks.append(d["camera_matrix"])
        Ds.append(d["dist_coeffs"])
        sizes.append(tuple(d["image_size"]))
    return Ks, Ds, sizes

def load_rig(path):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        return np.asarray(d["H_cam1_in_cam0"], float), np.asarray(d["H_cam2_in_cam0"], float)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return np.asarray(j["H_cam1_in_cam0"], float), np.asarray(j["H_cam2_in_cam0"], float)
    else:
        raise ValueError("Rig must be .npz or .json")

def _std_cols(cols): return [str(c).strip().lower().replace(" ", "") for c in cols]

def load_phantom_table(path, sheet=None):
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        import csv as _csv
        with open(path, newline="") as f:
            rdr = _csv.DictReader(f)
            cols = _std_cols(rdr.fieldnames or [])
            need = ["id","x","y","z"]
            if not all(c in cols for c in need):
                raise ValueError(f"{path} must have columns id,x,y,z (found {rdr.fieldnames})")
            idx_id = cols.index("id"); idx_x = cols.index("x"); idx_y = cols.index("y"); idx_z = cols.index("z")
            ids, xyz = [], []
            for row in rdr:
                vals = list(row.values())
                ids.append(int(vals[idx_id])); xyz.append([float(vals[idx_x]), float(vals[idx_y]), float(vals[idx_z])])
        return np.array(ids, np.int32), np.array(xyz, np.float64)
    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("Reading Excel requires pandas (pip install pandas).") from e
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
        if isinstance(df, dict):
            df = df[list(df.keys())[0]]
        cols = list(df.columns); cols_std = _std_cols(cols)
        need = ["id","x","y","z"]
        colmap = {}
        for req in need:
            if req not in cols_std:
                raise ValueError(f"{path} missing '{req}'. Found: {cols}")
            colmap[req] = cols[cols_std.index(req)]
        sub = df[[colmap["id"], colmap["x"], colmap["y"], colmap["z"]]].dropna()
        ids = sub[colmap["id"]].astype(int).to_numpy(np.int32)
        xyz = sub[[colmap["x"], colmap["y"], colmap["z"]]].astype(float).to_numpy(np.float64)
        return ids, xyz
    else:
        raise ValueError(f"Unsupported phantom file: {path}")

# ---------------- Two-click picker ----------------
class TwoClickPicker:
    def __init__(self, img, title="camX", roi_half=50, magnification=10, subpixel=True):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.title_main = f"{title} — L: pick ROI  |  q/Enter: accept  |  r: redo  |  Arrows: nudge  |  [ ]: nudge step"
        self.title_zoom = f"{title} — zoom (L: confirm, R/ESC: cancel, g: grid, sliders: threshold)"
        self.roi_half = int(roi_half)
        self.mag = int(magnification)
        self.subpixel = bool(subpixel)
        self.grid_on = True
        self.nudge_step = 0.10  # px

    def _clamp_roi(self, cx, cy):
        x0 = max(0, int(round(cx)) - self.roi_half)
        y0 = max(0, int(round(cy)) - self.roi_half)
        x1 = min(self.w-1, x0 + 2*self.roi_half)
        y1 = min(self.h-1, y0 + 2*self.roi_half)
        x0 = max(0, x1 - 2*self.roi_half)
        y0 = max(0, y1 - 2*self.roi_half)
        return x0, y0, x1, y1

    def _to_gray(self, im):
        return im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    def _ensure_u8(self, roi):
        if roi.dtype == np.uint8:
            return roi
        r = roi.astype(np.float32)
        r_min, r_max = float(r.min()), float(r.max())
        if r_max <= r_min:
            return np.zeros_like(roi, dtype=np.uint8)
        r = (r - r_min) * (255.0 / (r_max - r_min))
        return np.clip(r, 0, 255).astype(np.uint8)

    def _subpixel_refine(self, pt):
        if not self.subpixel:
            return float(pt[0]), float(pt[1])
        x, y = pt
        corners = np.array([[x, y]], dtype=np.float32).reshape(-1,1,2)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
        gray = self._to_gray(self.img)
        try:
            cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term)
        except cv2.error:
            pass
        c = corners.reshape(-1,2)[0]
        return float(c[0]), float(c[1])

    def _draw_grid(self, canvas, pane_origin=(0,0), roi_shape=None):
        if not self.grid_on or roi_shape is None:
            return
        H, W = roi_shape
        ox, oy = pane_origin
        step_roi = 5
        for i in range(0, W+1, step_roi):
            x = ox + i * self.mag
            cv2.line(canvas, (x, oy), (x, oy + H*self.mag), (80, 80, 80), 1, cv2.LINE_AA)
        for j in range(0, H+1, step_roi):
            y = oy + j * self.mag
            cv2.line(canvas, (ox, y), (ox + W*self.mag, y), (80, 80, 80), 1, cv2.LINE_AA)

    def _compose_zoom_view(self, roi_u8, mode, tval, invert, out_otzu_val=None):
        if mode == 0:
            otzu, bin_mask = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if out_otzu_val is not None:
                out_otzu_val[0] = int(otzu)
        else:
            _, bin_mask = cv2.threshold(roi_u8, int(tval), 255, cv2.THRESH_BINARY)

        if invert == 1:
            bin_mask = cv2.bitwise_not(bin_mask)

        zg = cv2.resize(roi_u8, (roi_u8.shape[1]*self.mag, roi_u8.shape[0]*self.mag), interpolation=cv2.INTER_NEAREST)
        zb = cv2.resize(bin_mask, (bin_mask.shape[1]*self.mag, bin_mask.shape[0]*self.mag), interpolation=cv2.INTER_NEAREST)
        zg = cv2.cvtColor(zg, cv2.COLOR_GRAY2BGR)
        zb = cv2.cvtColor(zb, cv2.COLOR_GRAY2BGR)

        H, W = roi_u8.shape[:2]
        cx, cy = (W*self.mag)//2, (H*self.mag)//2
        for pane in (zg, zb):
            cv2.drawMarker(pane, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

        if self.grid_on:
            self._draw_grid(zg, (0,0), (H,W))
            self._draw_grid(zb, (0,0), (H,W))

        vis = np.concatenate([zg, zb], axis=1)
        return vis, bin_mask

    def _zoom_pick(self, roi_gray):
        win = self.title_zoom
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        roi_u8 = self._ensure_u8(roi_gray)

        cv2.createTrackbar("Method (0=Auto,1=Manual)", win, 0, 1, lambda v: None)
        cv2.createTrackbar("Threshold",                 win, 128, 255, lambda v: None)
        cv2.createTrackbar("Invert (0/1)",             win, 0,   1,   lambda v: None)

        clicked = {"pt": None, "cancel": False}
        def mouse_cb(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked["pt"] = (x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                clicked["cancel"] = True
        cv2.setMouseCallback(win, mouse_cb)

        pane_w = roi_u8.shape[1] * self.mag
        otzu_val_holder = [None]

        while True:
            mode = cv2.getTrackbarPos("Method (0=Auto,1=Manual)", win)
            thr  = cv2.getTrackbarPos("Threshold", win)
            inv  = cv2.getTrackbarPos("Invert (0/1)", win)

            otzu_val_holder[0] = None
            vis, _ = self._compose_zoom_view(roi_u8, mode, thr, inv, out_otzu_val=otzu_val_holder)

            if mode == 0:
                label = f"Mode: AUTO (Otsu={otzu_val_holder[0] if otzu_val_holder[0] is not None else '?'}). Use Manual=1 for slider control."
            else:
                label = f"Mode: MANUAL (Thresh={thr})"
            hint = "L: confirm  |  R/ESC: cancel  |  g: grid on/off"
            cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 40), 2, cv2.LINE_AA)
            cv2.putText(vis, hint,  (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 40), 2, cv2.LINE_AA)

            cv2.imshow(win, vis)
            cv2.resizeWindow(win, min(vis.shape[1], 1600), min(vis.shape[0], 950))

            key = cv2.waitKey(10) & 0xFF
            if key == 27 or clicked["cancel"]:
                cv2.destroyWindow(win); return None
            if key in (ord('g'), ord('G')):
                self.grid_on = not self.grid_on
            if clicked["pt"] is not None:
                zx, zy = clicked["pt"]
                if zx >= pane_w:
                    zx -= pane_w
                rx = np.clip(zx / float(self.mag), 0.0, roi_u8.shape[1]-1.0)
                ry = np.clip(zy / float(self.mag), 0.0, roi_u8.shape[0]-1.0)
                cv2.destroyWindow(win)
                return (rx, ry)

    def run_once(self):
        cv2.namedWindow(self.title_main, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title_main, min(self.w, 1400), min(self.h, 900))

        base = self.img.copy()
        base_bgr = self._to_gray(base) if base.ndim == 2 else base
        if base_bgr.ndim == 2:
            base_bgr = cv2.cvtColor(base_bgr, cv2.COLOR_GRAY2BGR)

        chosen = {"pt": None}

        def main_mouse_cb(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                x0, y0, x1, y1 = self._clamp_roi(x, y)
                roi_full = self._to_gray(self.img)[y0:y1+1, x0:x1+1]
                res = self._zoom_pick(roi_full)
                if res is not None:
                    rx, ry = res
                    fx = x0 + rx
                    fy = y0 + ry
                    fx, fy = self._subpixel_refine((fx, fy))
                    chosen["pt"] = [fx, fy]

        cv2.setMouseCallback(self.title_main, main_mouse_cb)

        while True:
            vis = base_bgr.copy()
            cv2.putText(vis, f"Nudge step: {self.nudge_step:.3f}px   (Arrows/hjkl to nudge, [ ] adjust)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
            if chosen["pt"] is not None:
                cv2.circle(vis, (int(round(chosen["pt"][0])), int(round(chosen["pt"][1]))),
                           6, (0,255,255), 2)
            cv2.imshow(self.title_main, vis)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 13):
                if chosen["pt"] is not None:
                    p = (float(chosen["pt"][0]), float(chosen["pt"][1]))
                    cv2.destroyWindow(self.title_main)
                    return p
            elif key == ord('r'):
                chosen["pt"] = None
            elif chosen["pt"] is not None:
                if key in (81, ord('h')):
                    chosen["pt"][0] = max(0.0, chosen["pt"][0] - self.nudge_step)
                elif key in (83, ord('l')):
                    chosen["pt"][0] = min(self.w-1.0, chosen["pt"][0] + self.nudge_step)
                elif key in (82, ord('k')):
                    chosen["pt"][1] = max(0.0, chosen["pt"][1] - self.nudge_step)
                elif key in (84, ord('j')):
                    chosen["pt"][1] = min(self.h-1.0, chosen["pt"][1] + self.nudge_step)
                elif key == ord('['):
                    self.nudge_step = max(self.nudge_step/2.0, 0.01)
                elif key == ord(']'):
                    self.nudge_step = min(self.nudge_step*2.0, 5.0)

# ---------------- Triangulation & projection ----------------
def ray_from_pixel(uv, K, D):
    pts = np.array(uv, dtype=np.float64).reshape(1,1,2)
    norm = cv2.undistortPoints(pts, K, D)
    x, y = norm.reshape(2)
    d = np.array([x, y, 1.0], dtype=np.float64)
    d = d / np.linalg.norm(d)
    return d

def triangulate_three_rays(C_list, d_list):
    I = np.eye(3)
    A = np.zeros((3,3), dtype=np.float64)
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

def project_point_cam(X_world_in_cam0, K, D, R, t):
    rvec, _ = cv2.Rodrigues(R)
    obj = X_world_in_cam0.reshape(1,1,3).astype(np.float64)
    img, _ = cv2.projectPoints(obj, rvec, t.reshape(3,1), K, D)
    return img.reshape(2)

# ---------------- Rigid transforms ----------------
def kabsch(P, Q):
    """
    Find R,t (no scale) so that R*P + t ≈ Q.
    P: Nx3 in phantom frame, Q: Nx3 in cam0 frame
    Returns R_ph2c, t_ph2c (phantom->cam0).
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    assert P.shape == Q.shape and P.shape[1] == 3 and P.shape[0] >= 3
    cP = P.mean(axis=0); cQ = Q.mean(axis=0)
    X = (P - cP).T @ (Q - cQ)
    U,S,Vt = np.linalg.svd(X)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = cQ - R @ cP
    return R, t

def invert_rt(R, t):
    Ri = R.T
    ti = -Ri @ t
    return Ri, ti

# ---------------- Live capture helpers ----------------
DISPLAY_SCALE = 0.25
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 5472, 3648, 4
FOCUS_VALUE = None

def ensure_dirs(out_root):
    dirs = []
    for sub in ["left_captures","right_captures","back_captures"]:
        p = os.path.join(out_root, sub); os.makedirs(p, exist_ok=True); dirs.append(p)
    return dirs

def open_streams(cam_ids):
    try:
        from cam_stream import CameraStream
        streams = {}
        for cid in cam_ids:
            streams[cid] = CameraStream(device_index=cid, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS, focus_value=FOCUS_VALUE)
        return streams, True
    except Exception:
        streams = {}
        for cid in cam_ids:
            cap = cv2.VideoCapture(cid, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            streams[cid] = cap
        return streams, False

def get_frame(stream, use_custom):
    if use_custom:
        return stream.get_latest_frame()
    ret, frame = stream.read()
    return frame if ret else None

def release_streams(streams, use_custom):
    if use_custom:
        for s in streams.values(): s.stop()
    else:
        for cap in streams.values(): cap.release()
    cv2.destroyAllWindows()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Measure fiducials; auto-register to phantom after 3+ clicks. Reports in PHANTOM frame once possible.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", nargs=3, help="Explicit cam0, cam1, cam2 image paths")
    g.add_argument("--live", action="store_true", help="Open cameras and capture on SPACE")
    ap.add_argument("--cams", nargs=3, type=int, default=[0,1,2], help="Camera indices for live mode (cam0 cam1 cam2)")
    ap.add_argument("--out_root", default="captures", help="Where to save live captures")
    ap.add_argument("--intrinsics", nargs=3, required=True, help="intrin_cam_left.npz intrin_cam_right.npz intrin_cam_back.npz")
    ap.add_argument("--rig", required=True, help="rig_final.npz or .json")
    ap.add_argument("--phantom", required=True, help="phantom_points.xlsx or .csv with id,x,y,z (PHANTOM frame)")
    ap.add_argument("--phantom_sheet", default=None)
    ap.add_argument("--roi_half", type=int, default=50)
    ap.add_argument("--magnification", type=int, default=10)
    ap.add_argument("--no_subpixel", action="store_true")
    ap.add_argument("--csv_out", default="phantom_accuracy.csv")
    ap.add_argument("--anchors", nargs="+", type=int, default=[], help="Optional fixed anchor IDs for best‑fit; otherwise all measured so far are used")
    ap.add_argument("--ordered", action="store_true", help="Measure 1→47 in order (default is arbitrary IDs)")
    args = ap.parse_args()

    # Load calibration & phantom
    Ks, Ds, sizes = load_intrinsics(args.intrinsics)
    H01, H02 = load_rig(args.rig)
    R01, t01 = H01[:3,:3], H01[:3,3]
    R02, t02 = H02[:3,:3], H02[:3,3]
    phantom_ids, phantom_xyz = load_phantom_table(args.phantom, sheet=args.phantom_sheet)
    id2idx = {int(i):k for k,i in enumerate(phantom_ids)}
    def P(fid): return phantom_xyz[id2idx[int(fid)]]

    # Get images
    if args.images:
        save_paths = args.images
        stamp = os.path.splitext(os.path.basename(save_paths[0]))[0]
        if "_" in stamp: stamp = stamp.split("_",1)[1]
    else:
        out_dirs = ensure_dirs(args.out_root)
        streams, use_custom = open_streams(args.cams)
        frames = {}
        print("Live view: press SPACE to capture a set; 'q' to quit.")
        while True:
            for cid in args.cams:
                frm = get_frame(streams[cid], use_custom)
                if frm is None: continue
                frames[cid] = frm
                disp = cv2.resize(frm, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow(f"Cam{cid}", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                release_streams(streams, use_custom)
                print("Aborted."); return 1
            if k == 32:
                stamp = str(int(time.time()*1000))
                save_paths = []
                for i, cid in enumerate(args.cams):
                    CAM_NAMES = ["left", "right", "back"]
                    name = f"{CAM_NAMES[i]}_{stamp}.png"
                    path = os.path.join(out_dirs[i], name)
                    if cid in frames: cv2.imwrite(path, frames[cid]); save_paths.append(path)
                for cid in args.cams:
                    cv2.destroyWindow(f"Cam{cid}")
                release_streams(streams, use_custom)
                print("Captured:\n  " + "\n  ".join(save_paths))
                break

    # Load grayscale for picking
    g0 = cv2.imread(save_paths[0], cv2.IMREAD_GRAYSCALE)
    g1 = cv2.imread(save_paths[1], cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imread(save_paths[2], cv2.IMREAD_GRAYSCALE)
    if g0 is None or g1 is None or g2 is None:
        print("ERROR: failed to read images."); return 2

    # cam centers in cam0 frame
    C0 = np.zeros(3)
    C1 = -R01.T @ t01
    C2 = -R02.T @ t02
    # world->camera extrinsics for projection (cam0 frame is "world")
    Rw0, tw0 = np.eye(3), np.zeros(3)
    Rw1, tw1 = R01, t01
    Rw2, tw2 = R02, t02

    # helpers
    def click_triplet(label):
        p0 = TwoClickPicker(g0, title=f"cam0 — {label}", roi_half=args.roi_half, magnification=args.magnification, subpixel=(not args.no_subpixel)).run_once()
        if p0 is None: return None
        p1 = TwoClickPicker(g1, title=f"cam1 — {label}", roi_half=args.roi_half, magnification=args.magnification, subpixel=(not args.no_subpixel)).run_once()
        if p1 is None: return None
        p2 = TwoClickPicker(g2, title=f"cam2 — {label}", roi_half=args.roi_half, magnification=args.magnification, subpixel=(not args.no_subpixel)).run_once()
        if p2 is None: return None
        return (p0, p1, p2)

    def tri_from_pixels(p0, p1, p2):
        (u0,v0),(u1,v1),(u2,v2) = p0,p1,p2
        d0c = ray_from_pixel((u0,v0), Ks[0], Ds[0])
        d1c = ray_from_pixel((u1,v1), Ks[1], Ds[1])
        d2c = ray_from_pixel((u2,v2), Ks[2], Ds[2])
        d0 = d0c
        d1 = (R01.T @ d1c); d1 /= np.linalg.norm(d1)
        d2 = (R02.T @ d2c); d2 /= np.linalg.norm(d2)
        X = triangulate_three_rays([C0,C1,C2], [d0,d1,d2])
        pix0 = project_point_cam(X, Ks[0], Ds[0], Rw0, tw0)
        pix1 = project_point_cam(X, Ks[1], Ds[1], Rw1, tw1)
        pix2 = project_point_cam(X, Ks[2], Ds[2], Rw2, tw2)
        e0 = float(np.linalg.norm(pix0 - np.array([u0,v0])))
        e1 = float(np.linalg.norm(pix1 - np.array([u1,v1])))
        e2 = float(np.linalg.norm(pix2 - np.array([u2,v2])))
        rms = float(np.sqrt((e0*e0 + e1*e1 + e2*e2)/3.0))
        return X, (u0,v0,u1,v1,u2,v2), (e0,e1,e2,rms)

    # CSV header (match order!)
    header = ["timestamp","fid_id",
              "u0","v0","u1","v1","u2","v2",
              "X_cam","Y_cam","Z_cam",
              "X_meas_ph","Y_meas_ph","Z_meas_ph",
              "X_ref_ph","Y_ref_ph","Z_ref_ph",
              "dX","dY","dZ","err_norm",
              "err0_px","err1_px","err2_px","rms_px"]
    new_file = not os.path.isfile(args.csv_out)
    f = open(args.csv_out, "a", newline=""); writer = csv.writer(f)
    if new_file: writer.writerow(header)

    # Auto-registration state
    meas_cam = {}   # fid -> 3D in cam0
    ids_clicked = []

    def current_transform():
        """Compute PHANTOM->cam0 transform using all available anchors (>=3)."""
        if args.anchors:
            ids_for_fit = [fid for fid in args.anchors if fid in meas_cam]
        else:
            ids_for_fit = list(meas_cam.keys())
        ids_for_fit = sorted(set(ids_for_fit))
        if len(ids_for_fit) < 3:
            return None, None, ids_for_fit
        P_ph = np.vstack([P(fid) for fid in ids_for_fit])
        Q_c0 = np.vstack([meas_cam[fid] for fid in ids_for_fit])
        R_ph2c, t_ph2c = kabsch(P_ph, Q_c0)
        R_c2ph, t_c2ph = invert_rt(R_ph2c, t_ph2c)
        return (R_c2ph, t_c2ph), (R_ph2c, t_ph2c), ids_for_fit

    def measure_one(fid):
        clicks = click_triplet(f"fiducial {fid}")
        if clicks is None:
            print("Canceled."); return False
        X_cam, (u0,v0,u1,v1,u2,v2), (e0,e1,e2,rms) = tri_from_pixels(*clicks)

        # store
        meas_cam[fid] = X_cam.copy()
        ids_clicked.append(fid)

        # compute/refresh transform
        (R_c2ph, t_c2ph), (R_ph2c, t_ph2c), used_ids = current_transform()
        print(f"\nFID {fid:02d}")
        print(f"  Reproj RMS: {rms:.3f}   cam0:{e0:.3f}  cam1:{e1:.3f}  cam2:{e2:.3f}")
        if R_c2ph is None:
            print("  Not enough fiducials for comparison yet (need ≥3).")
            writer.writerow([time.time(), fid,
                             u0, v0, u1, v1, u2, v2,
                             X_cam[0], X_cam[1], X_cam[2],
                             "", "", "",
                             "", "", "",
                             "", "", "", "",
                             e0, e1, e2, rms])
            f.flush()
            return True

        # compare in PHANTOM frame
        X_meas_ph = R_c2ph @ X_cam + t_c2ph
        X_ref_ph  = P(fid)
        d = X_meas_ph - X_ref_ph
        err_norm = float(np.linalg.norm(d))
        print("  (PHANTOM frame reporting via on-the-fly best‑fit)")
        print(f"  Measured_ph:  [{X_meas_ph[0]:.3f}, {X_meas_ph[1]:.3f}, {X_meas_ph[2]:.3f}]")
        print(f"  Reference_ph: [{X_ref_ph[0]:.3f}, {X_ref_ph[1]:.3f}, {X_ref_ph[2]:.3f}]")
        print(f"  Δ = [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]   |Δ| = {err_norm:.3f}")
        print(f"  Fit used {len(used_ids)} anchor(s): {used_ids[:10]}{' ...' if len(used_ids)>10 else ''}")

        writer.writerow([time.time(), fid,
                         u0, v0, u1, v1, u2, v2,
                         X_cam[0], X_cam[1], X_cam[2],
                         X_meas_ph[0], X_meas_ph[1], X_meas_ph[2],
                         X_ref_ph[0], X_ref_ph[1], X_ref_ph[2],
                         d[0], d[1], d[2], err_norm,
                         e0, e1, e2, rms])
        f.flush()
        return True

    # -------------- Measurement loops --------------
    if args.ordered:
        print("\n=== Measure all 47 in order ===")
        print("For each fiducial, the picker will open in cam0 → cam1 → cam2.")
        print("If a fiducial is not visible, type 's' at the prompt to skip it.\n")
        for fid in range(1, 48):
            ans = input(f"Ready for fiducial {fid}? [Enter=go, s=skip, q=quit]: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                print(f"Skipping fiducial {fid}.")
                continue
            ok = measure_one(fid)
            if not ok:
                ans2 = input("Retry this fiducial? [Y/n]: ").strip().lower()
                if ans2 == "n":
                    print(f"Skipping fiducial {fid}.")
                    continue
                ok2 = measure_one(fid)
                if not ok2:
                    print(f"Skipping fiducial {fid} after second cancel.")
                    continue
        print("\nAll-done (ordered mode).")
    else:
        print("\n=== Arbitrary IDs mode ===")
        print("Enter fiducial ID 1–47 to measure, or 'q' to quit.")
        while True:
            s = input("Fiducial ID (1–47 or q): ").strip().lower()
            if s in ("q","quit","exit"): break
            if not s.isdigit():
                print("Please enter a number 1–47 or 'q'."); continue
            fid = int(s)
            if fid < 1 or fid > 47:
                print("Out of range. Enter 1–47."); continue
            measure_one(fid)

    f.close()
    print(f"\nSaved log to {args.csv_out}\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
