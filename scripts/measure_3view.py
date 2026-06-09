#!/usr/bin/env python3
# measure_distance_flexible_views.py — 2-or-3 view distance using rig, with per-camera skip
#
# Usage (same patterns as before):
#   python measure_distance_flexible_views.py \
#     --images cap0/cam0.png cap1/cam1.png cap2/cam2.png \
#     --rig rig_final.json --csv_out distances.csv --save_debug
#
#   python measure_distance_flexible_views.py \
#     --dirs capture0 capture1 capture2 --suffix 1759178742148 \
#     --rig rig_final.json
#
# Live capture remains available (SPACE to capture).
#
# NEW INTERACTION:
#   - When selecting each point on each camera view:
#       S  : skip this camera (mark as unreliable / no pick)
#       q/Enter: accept picked point
#       r  : redo
#       arrows/h j k l : nudge
#       [ ] : change nudge step
#       (Zoom window still supports Auto/Manual threshold, invert, grid)
#
# OUTPUT:
#   - Triangulates POINT A and POINT B from available views (2 or 3).
#   - CSV gains:
#       views_A_used, cams_A_used, baseline_A_deg, ...
#       views_B_used, cams_B_used, baseline_B_deg, caveats
#   - Reprojection RMS and per-cam pixel errors are computed only for used views.
#
# Caveats automatically warn when:
#   - Only 2 views were used (depth uncertainty higher).
#   - Baseline angle is small (< 5°: very poor; < 10°: weak).
#
import os, sys, json, csv, argparse, time
import numpy as np
import cv2

# ---------- IO helpers (from original file) ----------
def _resolve_rel(base_file, maybe_rel_path):
    if not maybe_rel_path:
        return None
    if os.path.isabs(maybe_rel_path):
        return maybe_rel_path
    return os.path.join(os.path.dirname(os.path.abspath(base_file)), maybe_rel_path)

def load_intrinsics_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["camera_matrix"], d["dist_coeffs"], tuple(d["image_size"])

def load_intrinsics_triplet(paths):
    Ks, Ds, sizes = [], [], []
    for p in paths:
        K, D, sz = load_intrinsics_npz(p)
        Ks.append(K); Ds.append(D); sizes.append(sz)
    return Ks, Ds, sizes

def load_rig(path):
    ext = os.path.splitext(path.lower())[1]
    labels = ["cam0","cam1","cam2"]
    intrin_paths = None
    if ext == ".json":
        with open(path,"r",encoding="utf-8") as f:
            j = json.load(f)
        H01 = np.asarray(j["H_cam1_in_cam0"], float)
        H02 = np.asarray(j["H_cam2_in_cam0"], float)
        if isinstance(j.get("cam_labels"), dict):
            labels = [j["cam_labels"].get("cam0","cam0"),
                      j["cam_labels"].get("cam1","cam1"),
                      j["cam_labels"].get("cam2","cam2")]
        if isinstance(j.get("intrinsics"), dict):
            intrin_paths = [
                _resolve_rel(path, j["intrinsics"].get("cam0","")),
                _resolve_rel(path, j["intrinsics"].get("cam1","")),
                _resolve_rel(path, j["intrinsics"].get("cam2","")),
            ]
        return H01, H02, labels, intrin_paths
    elif ext == ".npz":
        d = np.load(path, allow_pickle=True)
        H01 = np.asarray(d["H_cam1_in_cam0"], float)
        H02 = np.asarray(d["H_cam2_in_cam0"], float)
        return H01, H02, labels, None
    else:
        raise ValueError("Rig must be .npz or .json")

def label_short(s: str) -> str:
    return s[4:] if s.startswith("cam_") else s

# ---------- Advanced Two-click picker (extended with 'S' to skip) ----------
class TwoClickPicker:
    def __init__(self, img, title="camX", roi_half=50, magnification=10, subpixel=True, allow_skip=False):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.allow_skip = allow_skip
        self.title_main = f"{title} — L: pick ROI | S: skip | q/Enter: accept | r: redo | arrows/hjkl: nudge | [ ]: step"
        self.title_zoom = f"{title} — zoom (L: confirm, R/ESC: cancel, g: grid, Auto/Manual threshold)"
        self.roi_half = int(roi_half)
        self.mag = int(magnification)
        self.subpixel = bool(subpixel)
        self.grid_on = True
        self.nudge_step = 0.10  # px

    def _to_gray(self, im):
        if im.ndim == 2:
            return im
        return im[...,1]  # green channel

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

    def _clamp_roi(self, cx, cy):
        x0 = max(0, int(round(cx)) - self.roi_half)
        y0 = max(0, int(round(cy)) - self.roi_half)
        x1 = min(self.w-1, x0 + 2*self.roi_half)
        y1 = min(self.h-1, y0 + 2*self.roi_half)
        x0 = max(0, x1 - 2*self.roi_half)
        y0 = max(0, y1 - 2*self.roi_half)
        return x0, y0, x1, y1

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
                label = f"Mode: AUTO (Otsu={otzu_val_holder[0] if otzu_val_holder[0] is not None else '?'}). Manual=1 for slider."
            else:
                label = f"Mode: MANUAL (Thresh={thr})"
            hint = "L: confirm  |  R/ESC: cancel  |  g: grid"
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
                if zx >= pane_w: zx -= pane_w
                rx = np.clip(zx / float(self.mag), 0.0, roi_u8.shape[1]-1.0)
                ry = np.clip(zy / float(self.mag), 0.0, roi_u8.shape[0]-1.0)
                cv2.destroyWindow(win)
                return (rx, ry)

    def run_once(self):
        cv2.namedWindow(self.title_main, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title_main, min(self.w, 1400), min(self.h, 900))

        base_bgr = self.img.copy() if self.img.ndim == 3 else cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
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
            cv2.putText(vis, f"Nudge step: {self.nudge_step:.3f}px   (Arrows/hjkl nudge, [ ] step, r redo{', S skip' if self.allow_skip else ''})",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
            if chosen["pt"] is not None:
                cv2.circle(vis, (int(round(chosen["pt"][0])), int(round(chosen["pt"][1]))),
                           6, (0,255,255), 2)
            cv2.imshow(self.title_main, vis)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 13):  # accept
                if chosen["pt"] is not None:
                    p = (float(chosen["pt"][0]), float(chosen["pt"][1]))
                    cv2.destroyWindow(self.title_main)
                    return p
            elif key == ord('r'):
                chosen["pt"] = None
            elif self.allow_skip and key in (ord('s'), ord('S')):
                cv2.destroyWindow(self.title_main)
                return "SKIP"
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

# ---------- Geometry ----------
def ray_from_pixel(uv, K, D):
    pts = np.array(uv, dtype=np.float64).reshape(1,1,2)
    norm = cv2.undistortPoints(pts, K, D)
    x, y = norm.reshape(2)
    d = np.array([x, y, 1.0], dtype=np.float64)
    d = d / np.linalg.norm(d)
    return d

def triangulate_least_squares(C_list, d_list):
    """Least-squares point minimizing distances to all provided rays (>=2)."""
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

def angle_between(d1, d2):
    c = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def baseline_metrics(used_idx, d_rays):
    """Return maximum pairwise baseline angle among used rays (deg)."""
    if len(used_idx) < 2:
        return 0.0
    rays = [d_rays[i] for i in used_idx]
    angles = []
    for i in range(len(rays)):
        for j in range(i+1, len(rays)):
            angles.append(angle_between(rays[i], rays[j]))
    return float(max(angles)) if angles else 0.0

# ---------- Capture helpers ----------
DISPLAY_SCALE = 0.25
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 5472, 3648, 4
FOCUS_VALUE = None

def ensure_dirs(out_root):
    paths = []
    for sub in ["capture0", "capture1", "capture2"]:
        p = os.path.join(out_root, sub)
        os.makedirs(p, exist_ok=True)
        paths.append(p)
    return paths

def open_streams(cam_ids):
    try:
        from cam_stream import CameraStream
        streams = {}
        for cid in cam_ids:
            streams[cid] = CameraStream(device_index=cid, width=CAM_WIDTH, height=CAM_HEIGHT,
                                        fps=CAM_FPS, focus_value=FOCUS_VALUE)
        return streams, True
    except Exception as e:
        print(f"[Info] cam_stream unavailable ({e}); falling back to cv2.VideoCapture.")
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
        for s in streams.values():
            s.stop()
    else:
        for cap in streams.values():
            cap.release()
    cv2.destroyAllWindows()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Flexible 2- or 3-view point-to-point distance with per-camera skip.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", nargs=3, help="Explicit cam0, cam1, cam2 image paths")
    g.add_argument("--dirs", nargs=3, help="capture0 capture1 capture2 (use with --suffix)")
    g.add_argument("--live", action="store_true", help="Open cameras and capture on SPACE")

    ap.add_argument("--suffix", help="Common <STAMP> for camX_<STAMP>.png (required with --dirs)")
    ap.add_argument("--cams", nargs=3, type=int, default=[0,1,2], help="Device indices for --live (in rig cam order)")

    ap.add_argument("--rig", required=True, help="rig_final.json (preferred) or .npz")
    # Optional intrinsics overrides
    ap.add_argument("--intrinsics", nargs=3, help="Override intrinsics NPZs in rig cam order (cam0,cam1,cam2)")
    ap.add_argument("--intrin_rt", help="Override intrinsics for label cam_rt")
    ap.add_argument("--intrin_back", help="Override intrinsics for label cam_back")
    ap.add_argument("--intrin_lt", help="Override intrinsics for label cam_lt")

    ap.add_argument("--csv_out", default="distances.csv")
    ap.add_argument("--label", default="", help="Optional label for this measurement")
    ap.add_argument("--roi_half", type=int, default=40, help="Half-size of ROI in pixels (full image space)")
    ap.add_argument("--magnification", type=int, default=8, help="Zoom factor for the ROI window")
    ap.add_argument("--no_subpixel", action="store_true")
    ap.add_argument("--save_debug", action="store_true")
    args = ap.parse_args()

    # Load rig & labels
    H01, H02, labels, json_intrin = load_rig(args.rig)
    lbl_short = [label_short(s) for s in labels]
    print(f"[RIG] Labels: cam0={labels[0]}, cam1={labels[1]}, cam2={labels[2]}")

    # Resolve images
    if args.images:
        img_paths = args.images
        stamp = os.path.splitext(os.path.basename(img_paths[0]))[0]
        if "_" in stamp: stamp = stamp.split("_",1)[1]
    elif args.dirs:
        if not args.suffix:
            ap.error("--suffix is required when using --dirs")
        img_paths = [
            os.path.join(args.dirs[0], f"cam0_{args.suffix}.png"),
            os.path.join(args.dirs[1], f"cam1_{args.suffix}.png"),
            os.path.join(args.dirs[2], f"cam2_{args.suffix}.png"),
        ]
        stamp = args.suffix
    else:
        out_dirs = ensure_dirs("captures")
        streams, use_custom = open_streams(args.cams)
        frames = {}
        print("Live view: SPACE=capture, q=quit")
        while True:
            for idx, cid in enumerate(args.cams):
                frm = get_frame(streams[cid], use_custom)
                if frm is None: continue
                frames[cid] = frm
                disp = cv2.resize(frm, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow(f"{labels[idx]} ({cid})", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                release_streams(streams, use_custom)
                print("Aborted."); return 1
            if k == 32:  # SPACE
                stamp = str(int(time.time()*1000))
                img_paths = []
                for i, cid in enumerate(args.cams):
                    name = f"cam{i}_{stamp}.png"
                    path = os.path.join(out_dirs[i], name)
                    if cid in frames:
                        cv2.imwrite(path, frames[cid])
                        img_paths.append(path)
                for i in range(3):
                    try: cv2.destroyWindow(f"{labels[i]} ({args.cams[i]})")
                    except: pass
                release_streams(streams, use_custom)
                print("Captured:\n  " + "\n  ".join(img_paths))
                break

    for p in img_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"Image not found: {p}")

    # Intrinsics: precedence = label overrides > --intrinsics triple > rig JSON mapping
    intrin_paths = [None, None, None]
    if args.intrin_rt or args.intrin_back or args.intrin_lt:
        by_label = {"cam_rt": args.intrin_rt, "cam_back": args.intrin_back, "cam_lt": args.intrin_lt}
        for i, lab in enumerate(labels):
            if by_label.get(lab):
                intrin_paths[i] = by_label[lab]
    if args.intrinsics:
        for i in range(3):
            intrin_paths[i] = args.intrinsics[i]
    if any(p is None for p in intrin_paths):
        if json_intrin is not None:
            for i in range(3):
                if intrin_paths[i] is None:
                    intrin_paths[i] = json_intrin[i]
    if any(p is None for p in intrin_paths):
        raise SystemExit("Intrinsics not fully specified. Provide --intrinsics or label overrides, or a rig JSON with 'intrinsics'.")

    Ks, Ds, sizes = load_intrinsics_triplet(intrin_paths)
    print("[INTRIN]")
    for i in range(3):
        print(f"  {labels[i]}  <-  {os.path.basename(intrin_paths[i])}  size={sizes[i]}")

    # Extract extrinsics (cam0 is reference/world)
    R01, t01 = H01[:3,:3], H01[:3,3]
    R02, t02 = H02[:3,:3], H02[:3,3]
    C0 = np.zeros(3)
    C1 = -R01.T @ t01
    C2 = -R02.T @ t02
    Rw0, tw0 = np.eye(3), np.zeros(3)
    Rw1, tw1 = R01, t01
    Rw2, tw2 = R02, t02

    # Load images in COLOR
    c0 = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    c1 = cv2.imread(img_paths[1], cv2.IMREAD_COLOR)
    c2 = cv2.imread(img_paths[2], cv2.IMREAD_COLOR)
    if c0 is None or c1 is None or c2 is None:
        raise SystemExit("Failed to read one or more images.")

    def pick_point_flexible(name):
        # Returns tuple (pts, used_mask) where pts[i] is (u,v) or None; used_mask[i] is bool
        imgs = [c0, c1, c2]
        pts = [None, None, None]
        used = [False, False, False]
        for i in range(3):
            picker = TwoClickPicker(imgs[i],
                                    title=f"{name} — {labels[i]} (S=skip)",
                                    roi_half=args.roi_half,
                                    magnification=args.magnification,
                                    subpixel=(not args.no_subpixel),
                                    allow_skip=True)
            res = picker.run_once()
            if res == "SKIP":
                print(f"[{name}] Skipped {labels[i]}")
                continue
            if res is None:
                print(f"[{name}] Canceled.")
                return None, None
            pts[i] = (float(res[0]), float(res[1]))
            used[i] = True
        if sum(used) < 2:
            print(f"[{name}] Need at least TWO cameras. You used {sum(used)}.")
            return None, None
        return pts, used

    def triangulate_from_optional_pixels(pts, used):
        # Build rays and centers for used views in cam0 frame
        d_cam = [None, None, None]
        C_cam = [C0, C1, C2]
        # d0 from cam0 pixels
        if used[0]:
            d_cam[0] = ray_from_pixel(pts[0], Ks[0], Ds[0])
        # cam1/cam2 rays transformed into cam0 frame
        if used[1]:
            d1c = ray_from_pixel(pts[1], Ks[1], Ds[1])
            d_cam[1] = (R01.T @ d1c); d_cam[1] /= np.linalg.norm(d_cam[1])
        if used[2]:
            d2c = ray_from_pixel(pts[2], Ks[2], Ds[2])
            d_cam[2] = (R02.T @ d2c); d_cam[2] /= np.linalg.norm(d_cam[2])

        used_idx = [i for i,v in enumerate(used) if v]
        C_list = [C_cam[i] for i in used_idx]
        d_list = [d_cam[i] for i in used_idx]

        X = triangulate_least_squares(C_list, d_list)

        # Reprojection only for used cameras
        proj = [None, None, None]
        err_each = [None, None, None]
        for i in used_idx:
            if i == 0:
                pix = project_point_cam(X, Ks[0], Ds[0], Rw0, tw0)
            elif i == 1:
                pix = project_point_cam(X, Ks[1], Ds[1], Rw1, tw1)
            else:
                pix = project_point_cam(X, Ks[2], Ds[2], Rw2, tw2)
            proj[i] = pix
            err_each[i] = float(np.linalg.norm(pix - np.array(pts[i])))

        # RMS over used views
        used_errs = [e for e in err_each if e is not None]
        rms = float(np.sqrt(np.mean(np.square(used_errs)))) if used_errs else float("nan")

        # Baseline metric
        baseline_deg = baseline_metrics(used_idx, d_cam)

        return X, proj, err_each, rms, used_idx, baseline_deg

    # CSV header (extended)
    stamp = os.path.splitext(os.path.basename(img_paths[0]))[0]
    if "_" in stamp: stamp = stamp.split("_",1)[1]
    L0, L1, L2 = lbl_short[0], lbl_short[1], lbl_short[2]
    header = ["stamp","label",
              f"u_{L0}_A", f"v_{L0}_A", f"u_{L1}_A", f"v_{L1}_A", f"u_{L2}_A", f"v_{L2}_A",
              f"u_{L0}_B", f"v_{L0}_B", f"u_{L1}_B", f"v_{L1}_B", f"u_{L2}_B", f"v_{L2}_B",
              "XA","YA","ZA","XB","YB","ZB",
              "dX","dY","dZ","distance",
              f"A_err_{L0}_px", f"A_err_{L1}_px", f"A_err_{L2}_px", "A_rms_px",
              f"B_err_{L0}_px", f"B_err_{L1}_px", f"B_err_{L2}_px", "B_rms_px",
              "views_A_used", "cams_A_used", "baseline_A_deg",
              "views_B_used", "cams_B_used", "baseline_B_deg",
              "caveats"]
    new_file = not os.path.isfile(args.csv_out)
    f = open(args.csv_out, "a", newline="")
    writer = csv.writer(f)
    if new_file: writer.writerow(header)

    # --- POINT A ---
    print(f"\nSelect POINT A — {labels[0]} → {labels[1]} → {labels[2]} (press S on any camera to skip)")
    ptsA, usedA = pick_point_flexible("POINT A")
    if ptsA is None:
        f.close(); sys.exit("Canceled at POINT A.")
    XA, projA, errA_each, errA_rms, usedA_idx, baseA = triangulate_from_optional_pixels(ptsA, usedA)

    # --- POINT B ---
    print(f"\nSelect POINT B — {labels[0]} → {labels[1]} → {labels[2]} (press S on any camera to skip)")
    ptsB, usedB = pick_point_flexible("POINT B")
    if ptsB is None:
        f.close(); sys.exit("Canceled at POINT B.")
    XB, projB, errB_each, errB_rms, usedB_idx, baseB = triangulate_from_optional_pixels(ptsB, usedB)

    # Distance
    dvec = XB - XA
    dist = float(np.linalg.norm(dvec))

    # Build caveats
    caveats = []
    def cams_str(idx_list): return ",".join([lbl_short[i] for i in idx_list])
    if len(usedA_idx) == 2 or len(usedB_idx) == 2:
        caveats.append("Only 2 views used for at least one point; depth uncertainty increases (elongation along epipolar).")
    for tag, base in (("A", baseA), ("B", baseB)):
        if base < 5.0:
            caveats.append(f"Very small baseline angle for {tag} (~{base:.1f}°): distance may be unreliable.")
        elif base < 10.0:
            caveats.append(f"Weak baseline angle for {tag} (~{base:.1f}°): expect higher uncertainty.")
    caveat_text = " ".join(caveats) if caveats else ""

    # Print summary
    print("\n=== RESULT ===")
    print(f"XA = [{XA[0]:.3f}, {XA[1]:.3f}, {XA[2]:.3f}]  (cam0 frame)")
    print(f"XB = [{XB[0]:.3f}, {XB[1]:.3f}, {XB[2]:.3f}]  (cam0 frame)")
    print(f"Δ  = [{dvec[0]:.3f}, {dvec[1]:.3f}, {dvec[2]:.3f}]  | Distance = {dist:.3f}")
    print(f"A reproj RMS = {errA_rms:.2f} px over cams {cams_str(usedA_idx)}  (per-cam: {[None if e is None else round(e,2) for e in errA_each]})")
    print(f"B reproj RMS = {errB_rms:.2f} px over cams {cams_str(usedB_idx)}  (per-cam: {[None if e is None else round(e,2) for e in errB_each]})")
    print(f"Baseline angles (max pairwise): A={baseA:.2f}°, B={baseB:.2f}°")
    if caveat_text:
        print(f"Caveats: {caveat_text}")

    # Save debug overlays (optional; only for used cams)
    if args.save_debug:
        for tag, pts, projs, used_mask in [("A", ptsA, projA, usedA), ("B", ptsB, projB, usedB)]:
            for i in range(3):
                if not used_mask[i]: continue
                src = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
                if src is None: continue
                uv_click = pts[i]
                pix_proj = projs[i]
                # clicked (blue)
                cv2.circle(src, (int(round(uv_click[0])), int(round(uv_click[1]))), 7, (255,0,0), 2)
                # reprojected (red cross)
                cv2.drawMarker(src, (int(round(pix_proj[0])), int(round(pix_proj[1]))),
                               (0,0,255), markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=16, thickness=2)
                outpng = f"dist_debug_{tag}_{labels[i]}_{stamp}.png"
                cv2.imwrite(outpng, src)

    # Helper to write u,v in CSV (blank if not used)
    def uv_or_blank(pts, i):
        if pts[i] is None: return ("", "")
        return (pts[i][0], pts[i][1])

    # Log to CSV
    row = [
        stamp, args.label,
        *uv_or_blank(ptsA,0), *uv_or_blank(ptsA,1), *uv_or_blank(ptsA,2),
        *uv_or_blank(ptsB,0), *uv_or_blank(ptsB,1), *uv_or_blank(ptsB,2),
        XA[0], XA[1], XA[2], XB[0], XB[1], XB[2],
        dvec[0], dvec[1], dvec[2], dist,
        # per-cam errors (blank if not used)
        "" if errA_each[0] is None else errA_each[0],
        "" if errA_each[1] is None else errA_each[1],
        "" if errA_each[2] is None else errA_each[2],
        errA_rms,
        "" if errB_each[0] is None else errB_each[0],
        "" if errB_each[1] is None else errB_each[1],
        "" if errB_each[2] is None else errB_each[2],
        errB_rms,
        len(usedA_idx), cams_str(usedA_idx), baseA,
        len(usedB_idx), cams_str(usedB_idx), baseB,
        caveat_text
    ]
    writer.writerow(row)
    f.flush(); f.close()
    print(f"\nSaved to {args.csv_out}")
    print("Done.")

if __name__ == "__main__":
    sys.exit(main())
