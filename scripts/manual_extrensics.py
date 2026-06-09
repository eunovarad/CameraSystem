#!/usr/bin/env python3
# manual_extrinsics.py
#
# Two-click fiducial picking with magnified ROI for precise center selection.
# - First click: choose ROI on the full image
# - Second click: pick precise center in the zoomed ROI
# - Optional subpixel refinement around the precise click
#
# Then solves PnP per camera, outputs rig and reprojection overlays.
#
# Usage A (point to specific images):
#   python manual_extrinsics.py ^
#     --images capture0/cam0_1755551159870.png capture1/cam1_1755551159870.png capture2/cam2_1755551159870.png ^
#     --intrinsics intrinsics_cam0.npz intrinsics_cam1.npz intrinsics_cam2.npz ^
#     --phantom phantom_points.xlsx --save_debug
#
# Usage B (dirs + common suffix):
#   python manual_extrinsics.py ^
#     --dirs capture0 capture1 capture2 --suffix 1755551159870 ^
#     --intrinsics intrinsics_cam0.npz intrinsics_cam1.npz intrinsics_cam2.npz ^
#     --phantom phantom_points.xlsx --save_debug
#
# Controls (main image window):
#   - Left click: choose ROI (opens zoom window)
#   - Right click: undo last accepted point
#   - 'q': finish current camera
#   - 'c': clear all points for current camera
#   - Arrows or h/j/k/l: nudge the last point
#   - '[' and ']': adjust nudge step
#
# Controls (zoom window):
#   - Left click: confirm precise point (closes zoom window)
#   - Right click or 'esc': cancel this ROI (do not add a point)
#
import os, sys, json, csv, argparse
import numpy as np
import cv2

# ----------------- IO helpers -----------------
def load_intrinsics(paths):
    Ks, Ds, sizes = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Ks.append(d["camera_matrix"])
        Ds.append(d["dist_coeffs"])
        sizes.append(tuple(d["image_size"]))  # (w,h)
    return Ks, Ds, sizes

def _std_cols(cols): return [str(c).strip().lower().replace(" ", "") for c in cols]

def load_phantom_table(path, sheet=None):
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
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
    elif ext in (".xlsx",".xls"):
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("Reading Excel requires pandas (pip install pandas).") from e
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
        if isinstance(df, dict):
            df = df[list(df.keys())[0]]
        cols = list(df.columns)
        cols_std = _std_cols(cols)
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

# ----------------- Two-click picker -----------------
class TwoClickPicker:
    """
    Two-click ROI→zoom picker with:
      - Live thresholding (Auto Otsu / Manual + slider, invert)
      - Side-by-side zoom (gray | binary), grid toggle (g)
      - Keyboard nudge in main view (arrows / h j k l), '[' and ']' adjust step
      - Robust slider handling via per-frame polling
      - 16-bit ROI safe (auto-normalizes to 8-bit for display/thresholding)

    Public API:
        run_once() -> (x, y)          # pick one point
        run()      -> Nx2 float array  # pick many points; right-click undo; 'q' finish
    """
    def __init__(self, img, title="camX", roi_half=50, magnification=10, subpixel=True):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.title_main = f"{title} — L: pick ROI  |  q: finish  |  r: redo sel  |  Arrows/hjkl: nudge  |  [ ]: step"
        self.title_zoom = f"{title} — zoom (L: confirm, R/ESC: cancel, g: grid, sliders: threshold)"
        self.roi_half = int(roi_half)
        self.mag = int(magnification)
        self.subpixel = bool(subpixel)
        self.grid_on = True
        self.nudge_step = 0.10  # px

    # ---------- helpers ----------
    def _clamp_roi(self, cx, cy):
        x0 = max(0, int(round(cx)) - self.roi_half)
        y0 = max(0, int(round(cy)) - self.roi_half)
        x1 = min(self.w-1, x0 + 2*self.roi_half)
        y1 = min(self.h-1, y0 + 2*self.roi_half)
        x0 = max(0, x1 - 2*self.roi_half)
        y0 = max(0, y1 - 2*self.roi_half)
        return x0, y0, x1, y1

    def _to_gray(self, im):
        if im.ndim == 2:
            return im
        return im[...,1]

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

    # ---------- main public APIs ----------
    def run_once(self):
        """Pick one point with ROI→zoom workflow. Returns (x,y) in full-image coords."""
        cv2.namedWindow(self.title_main, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title_main, min(self.w, 1400), min(self.h, 900))

        base_bgr = self._to_gray(self.img) if self.img.ndim == 2 else self.img
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
            cv2.putText(vis, f"Nudge step: {self.nudge_step:.3f}px   (Arrows/hjkl to nudge, [ ] to adjust)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
            if chosen["pt"] is not None:
                cv2.circle(vis, (int(round(chosen["pt"][0])), int(round(chosen["pt"][1]))),
                           6, (0,255,255), 2)
            cv2.imshow(self.title_main, vis)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 13):  # q or Enter -> accept and return point
                if chosen["pt"] is not None:
                    p = (float(chosen["pt"][0]), float(chosen["pt"][1]))
                    cv2.destroyWindow(self.title_main)
                    return p
            elif key == ord('r'):
                chosen["pt"] = None
            elif chosen["pt"] is not None:
                if key in (81, ord('h')):   # left
                    chosen["pt"][0] = max(0.0, chosen["pt"][0] - self.nudge_step)
                elif key in (83, ord('l')): # right
                    chosen["pt"][0] = min(self.w-1.0, chosen["pt"][0] + self.nudge_step)
                elif key in (82, ord('k')): # up
                    chosen["pt"][1] = max(0.0, chosen["pt"][1] - self.nudge_step)
                elif key in (84, ord('j')): # down
                    chosen["pt"][1] = min(self.h-1.0, chosen["pt"][1] + self.nudge_step)
                elif key == ord('['):
                    self.nudge_step = max(self.nudge_step/2.0, 0.01)
                elif key == ord(']'):
                    self.nudge_step = min(self.nudge_step*2.0, 5.0)

    def run(self):
        """
        Multi-point picker.
        - Left click -> ROI/zoom confirm to add a point
        - Right click -> undo last point
        - 'c' -> clear all
        - Arrows/h/j/k/l -> nudge last point
        - '[' and ']' -> adjust nudge step
        - 'q' -> finish and return Nx2 array
        """
        cv2.namedWindow(self.title_main, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title_main, min(self.w, 1400), min(self.h, 900))

        gray = self._to_gray(self.img)
        base_bgr = self.img.copy() if self.img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        points = []  # list of [x, y]

        def main_mouse_cb(event, x, y, flags, userdata):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                x0, y0, x1, y1 = self._clamp_roi(x, y)
                roi_full = gray[y0:y1+1, x0:x1+1]
                res = self._zoom_pick(roi_full)
                if res is not None:
                    rx, ry = res
                    fx = x0 + rx
                    fy = y0 + ry
                    fx, fy = self._subpixel_refine((fx, fy))
                    points.append([fx, fy])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if points:
                    points.pop()

        cv2.setMouseCallback(self.title_main, main_mouse_cb)

        last_nudge_index = -1  # track which point is being nudged (last one)
        while True:
            vis = base_bgr.copy()
            # draw existing points
            for i, (px, py) in enumerate(points, start=1):
                cv2.circle(vis, (int(round(px)), int(round(py))), 5, (0,255,255), 2)
                cv2.putText(vis, f"{i}", (int(round(px))+18, int(round(py))-18), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 3, cv2.LINE_AA)

            msg1 = f"pts: {len(points)}   Nudge step: {self.nudge_step:.3f}px"
            msg2 = "L: add  |  R: undo  |  c: clear  |  q: finish  |  arrows/hjkl: nudge last  |  [ ]: step"
            cv2.putText(vis, msg1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,220,20), 2, cv2.LINE_AA)
            cv2.putText(vis, msg2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,220,20), 2, cv2.LINE_AA)

            cv2.imshow(self.title_main, vis)
            key = cv2.waitKeyEx(20) & 0xFF

            if key in (ord('q'), ord('Q'), 13, 10):
                arr = np.array(points, dtype=np.float32) if points else np.zeros((0,2), np.float32)
                # keep window open for ID entry
                return arr
            elif key == ord('c'):
                points.clear()
                last_nudge_index = -1
            elif key == ord('['):
                self.nudge_step = max(self.nudge_step/2.0, 0.01)
            elif key == ord(']'):
                self.nudge_step = min(self.nudge_step*2.0, 5.0)
            elif points:
                # nudge the last point
                i = len(points) - 1
                px, py = points[i]
                if key in (81, ord('h')):   # left
                    px = max(0.0, px - self.nudge_step)
                elif key in (83, ord('l')): # right
                    px = min(self.w-1.0, px + self.nudge_step)
                elif key in (82, ord('k')): # up
                    py = max(0.0, py - self.nudge_step)
                elif key in (84, ord('j')): # down
                    py = min(self.h-1.0, py + self.nudge_step)
                else:
                    continue
                points[i] = [px, py]
                last_nudge_index = i

# ----------------- SE(3) helpers -----------------
def rvec_tvec_to_H(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    H = np.eye(4); H[:3,:3] = R; H[:3,3] = tvec.reshape(3)
    return H

def invert_H(H):
    R = H[:3,:3]; t = H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = R.T; Hi[:3,3] = -R.T @ t
    return Hi

# ----------------- PnP from manual clicks -----------------
def solve_extrinsics_from_clicks(img_path, K, D, phantom_ids, phantom_xyz,
                                 min_points=12, save_debug=False, cam_index=None,
                                 roi_half=40, magnification=8, subpixel=True):
    # load image
    src_full = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if src_full is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    # two-click collector (keep color main view; ROI gray on demand)
    picker = TwoClickPicker(src_full, title=os.path.basename(img_path),
                            roi_half=roi_half, magnification=magnification, subpixel=subpixel)

    # >>> FIX: use picker.run() to collect MANY points
    clicks = picker.run()  # Nx2 float array
    if clicks.shape[0] < min_points:
        raise RuntimeError(f"Only {clicks.shape[0]} points clicked (<{min_points}).")

        # Show list and ask IDs (single robust loop)
    print("")
    print(f"Clicked {len(clicks)} points on {os.path.basename(img_path)}.")
    for i,(x,y) in enumerate(clicks, start=1):
        print(f"  {i:02d}: x={x:.2f}, y={y:.2f}")
    print("")
    print("Enter phantom IDs in the SAME ORDER as clicked (comma/space separated). Type 'r' to redo picking for this image.")
    while True:
        id_str = input("IDs: ").strip()
        if id_str.lower() in ("r","redo"):
            print("[INFO] Redo requested. Re-opening picker for this image...")
            try:
                cv2.destroyWindow(picker.title_main)
            except Exception:
                pass
            try:
                cv2.destroyWindow(picker.title_zoom)
            except Exception:
                pass
            cv2.waitKey(1)
            picker = TwoClickPicker(src_full, title=f"{os.path.basename(img_path)} — L: pick ROI | q: finish | right: undo | arrows: nudge")
            clicks = picker.run()
            print(f"[INFO] Re-captured {len(clicks)} point(s).")
            print("Re-enter phantom IDs in the SAME ORDER as clicked (comma/space). Type 'r' to redo again.")
            continue
        try:
            import re as _re
            nums = _re.findall(r"-?\d+", id_str)
            id_list = [int(n) for n in nums]
        except Exception:
            print("[WARN] Could not parse your entry. Try again or type 'r' to redo picking.")
            continue
        if len(id_list) != len(clicks):
            print(f"[WARN] You entered {len(id_list)} IDs, but clicked {len(clicks)} points. Re-enter or type 'r' to redo.")
            continue
        # Success: close windows now and proceed
        try:
            cv2.destroyWindow(picker.title_main)
        except Exception:
            pass
        try:
            cv2.destroyWindow(picker.title_zoom)
        except Exception:
            pass
        for _ in range(3):
            cv2.waitKey(10)
        print(f"[INFO] IDs accepted ({len(id_list)} for {len(clicks)} clicks): {id_list}. Closing picker and continuing...")
        break
# Map IDs → XYZ
    id_to_idx = {int(i):k for k,i in enumerate(phantom_ids)}
    try:
        idxs = [id_to_idx[i] for i in id_list]
    except KeyError as e:
        raise RuntimeError(f"Phantom ID {e} not in phantom table.") from None

    obj = phantom_xyz[idxs].astype(np.float64)       # Nx3
    img = clicks.astype(np.float64).reshape(-1,1,2)  # Nx1x2

    # ----- Robust PnP: undistort 2D points, RANSAC seed + LM refine -----
    img_pts = clicks.astype(np.float64).reshape(-1,1,2)
    und = cv2.undistortPoints(img_pts, K, D, P=K)
    und_pts = und.reshape(-1,1,2)

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, und_pts, K, None,
        iterationsCount=1000, reprojectionError=1.5, confidence=0.999,
        flags=cv2.SOLVEPNP_AP3P
    )
    if not ok or rvec is None:
        ok2, rvec2, tvec2 = cv2.solvePnP(obj, und_pts, K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok2:
            raise RuntimeError("PnP (RANSAC and EPnP) failed.")
        rvec, tvec = rvec2, tvec2

    if isinstance(inl, np.ndarray) and inl.size >= max(6, int(0.6*len(obj))):
        mask = np.zeros(len(obj), bool); mask[inl.reshape(-1)] = True
        obj_use, und_use = obj[mask], und_pts[mask]
        img_use = img_pts[mask]
    else:
        obj_use, und_use = obj, und_pts
        img_use = img_pts

    rvec, tvec = cv2.solvePnPRefineLM(obj_use, und_use, K, None, rvec, tvec)

    # RMS in original pixel space (with distortion)
    reproj, _ = cv2.projectPoints(obj_use, rvec, tvec, K, D)
    reproj = reproj.reshape(-1,2)
    err = np.sqrt(np.sum((reproj - img_use.reshape(-1,2))**2, axis=1))
    rms = float(np.sqrt(np.mean(err**2)))
    print(f"PnP complete. Inliers used: {len(obj_use)}/{len(obj)}  RMS: {rms:.2f} px")


    # overlay
    if save_debug:
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        for (x,y) in clicks:
            cv2.circle(src, (int(round(x)),int(round(y))), 6, (255,0,0), 2)
        for (u,v) in reproj:
            cv2.drawMarker(src, (int(round(u)),int(round(v))), (0,0,255),
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
        CAM_NAMES = ["left", "right", "back"]
        tag = CAM_NAMES[cam_index] if cam_index is not None else "camX"
        out_png = f"manual_reproj_{tag}.png"
        cv2.imwrite(out_png, src)
        print(f"Saved overlay: {out_png}")

    H_c_w = rvec_tvec_to_H(rvec, tvec)
    return H_c_w, rvec, tvec, rms, clicks, id_list

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Two-click (ROI + zoom) manual extrinsics from 3D phantom.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", nargs=3, help="Explicit image paths for cam0, cam1, cam2")
    g.add_argument("--dirs", nargs=3, help="Folders for cam0, cam1, cam2; use with --suffix")
    ap.add_argument("--suffix", help="Common suffix (the <STAMP>) to pick camX_<STAMP> in each folder")
    ap.add_argument("--intrinsics", nargs=3, required=True, help="intrin_cam_left.npz intrin_cam_right.npz intrin_cam_back.npz")
    ap.add_argument("--phantom", required=True, help="phantom_points.xlsx or .csv with columns id,x,y,z")
    ap.add_argument("--phantom_sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--min_points", type=int, default=12, help="Minimum clicks/IDs per camera")
    ap.add_argument("--save_debug", action="store_true", help="Save reprojection overlays")
    ap.add_argument("--json_out", default="rig_from_clicks.json", help="JSON output")
    ap.add_argument("--npz_out", default="rig_from_clicks.npz", help="NPZ output")
    # ROI/zoom controls
    ap.add_argument("--roi_half", type=int, default=40, help="Half-size of ROI in pixels (full image space)")
    ap.add_argument("--magnification", type=int, default=8, help="Zoom factor for the ROI window")
    ap.add_argument("--no_subpixel", action="store_true", help="Disable subpixel refine on the 2nd click")
    args = ap.parse_args()

    # Resolve images
    if args.images:
        img_paths = args.images
    else:
        if not args.suffix:
            ap.error("--suffix is required when using --dirs")
        img_paths = [
            os.path.join(args.dirs[0], f"left_{args.suffix}.png"),
            os.path.join(args.dirs[1], f"right_{args.suffix}.png"),
            os.path.join(args.dirs[2], f"back_{args.suffix}.png"),
        ]
    for p in img_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")

    # Load intrinsics + phantom
    Ks, Ds, sizes = load_intrinsics(args.intrinsics)
    phantom_ids, phantom_xyz = load_phantom_table(args.phantom, sheet=args.phantom_sheet)

    # Per-camera manual solve
    results = []
    for cam in range(3):
        print("\n" + "="*78)
        print(f"Camera {cam}: {img_paths[cam]}")
        print("="*78)
                # Scale K to current image size if needed
        im0 = cv2.imread(img_paths[cam], cv2.IMREAD_UNCHANGED)
        Himg, Wimg = im0.shape[:2]
        cw, ch = sizes[cam]  # (w,h) from intrinsics
        Kc = Ks[cam].copy()
        if (Wimg, Himg) != (int(cw), int(ch)):
            sx, sy = Wimg/float(cw), Himg/float(ch)
            Kc[0,0] *= sx; Kc[1,1] *= sy
            Kc[0,2] *= sx; Kc[1,2] *= sy
            print(f"[INFO] Scaled K for cam {cam}: sx={sx:.6f}, sy={sy:.6f}")

        H, rvec, tvec, rms, clicks, id_list = solve_extrinsics_from_clicks(
            img_paths[cam], Kc, Ds[cam],
            phantom_ids, phantom_xyz,
            min_points=args.min_points,
            save_debug=args.save_debug,
            cam_index=cam,
            roi_half=args.roi_half,
            magnification=args.magnification,
            subpixel=(not args.no_subpixel)
        )
        results.append((H, rvec, tvec, rms, clicks, id_list))

    # Build rig transforms
    H0, H1, H2 = results[0][0], results[1][0], results[2][0]
    H01 = H1 @ invert_H(H0)   # cam1 in cam0 frame
    H02 = H2 @ invert_H(H0)   # cam2 in cam0 frame

    # Save NPZ
    np.savez(args.npz_out,
             H_cam0=H0, H_cam1=H1, H_cam2=H2,
             H_cam1_in_cam0=H01, H_cam2_in_cam0=H02,
             rvec_cam0=results[0][1], tvec_cam0=results[0][2], rms_cam0=results[0][3],
             rvec_cam1=results[1][1], tvec_cam1=results[1][2], rms_cam1=results[1][3],
             rvec_cam2=results[2][1], tvec_cam2=results[2][2], rms_cam2=results[2][3],
             intrinsics_cam0=args.intrinsics[0], intrinsics_cam1=args.intrinsics[1], intrinsics_cam2=args.intrinsics[2],
             images=np.array(img_paths),
             phantom=args.phantom)
    print(f"\nWrote {args.npz_out}")

    # Save JSON
    def tolist(M): return np.asarray(M, dtype=float).tolist()
    out = {
        "frames": {"reference":"cam0"},
        "images": img_paths,
        "intrinsics": {"cam0": args.intrinsics[0], "cam1": args.intrinsics[1], "cam2": args.intrinsics[2]},
        "H_cam0": tolist(H0), "H_cam1": tolist(H1), "H_cam2": tolist(H2),
        "H_cam1_in_cam0": tolist(H01), "H_cam2_in_cam0": tolist(H02),
        "reprojection_rms_px": {"cam0": float(results[0][3]), "cam1": float(results[1][3]), "cam2": float(results[2][3])},
        "phantom_file": args.phantom
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.json_out}")

    print("\nDone. Review the PNG overlays (if --save_debug) for millimetric alignment."
          "\nTip: If RMS >2–3 px, add more well-spread points or re-click ambiguous ones.")

if __name__ == "__main__":
    sys.exit(main())
