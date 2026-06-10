#!/usr/bin/env python3
# =========================
#   SIMPLE PRESET CONFIG
# =========================

USE_PRESET = True

CAM = "right"   # change to "left", "right", or "back"            #********************* change this when you switch cameras

PRESET = {
    "glob": [f"./data/{CAM}_captures/*.png"],
    "intrinsics": f"intrin_cam_{CAM}.npz",
    "rows": 7,
    "cols": 11,
    "square": 20.0,
    "save_overlay": True,
    "overlay_dir": f"./data/{CAM}_overlays",
    "verbose": True
}

import argparse, glob, os, sys, csv
import numpy as np
import cv2

def npz_list_keys(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print(f"[INFO] Keys in {npz_path}:")
    for k in data.files:
        v = data[k]
        try:
            shape = v.shape; dtype = v.dtype
        except Exception:
            shape = None; dtype = None
        print(f"  - {k}: shape={shape}, dtype={dtype}")
    return 0

def _first_present(npz, keys):
    for k in keys:
        if k in npz: return npz[k]
    return None

def _to_size_tuple(x):
    if x is None: return None
    try:
        arr = np.array(x).reshape(-1)
        if arr.size >= 2: return (int(arr[0]), int(arr[1]))
    except Exception: pass
    return None

def _pick_size_orientation(saved_size_wh, img_wh):
    if saved_size_wh is None: return None
    w_img, h_img = img_wh
    if w_img <= 0 or h_img <= 0: return saved_size_wh
    asp_img = float(w_img) / float(h_img)
    w0, h0 = saved_size_wh
    cand = [(w0,h0),(h0,w0)]
    def delta(a,b): return abs(float(a)/float(b) - asp_img)
    return min(cand, key=lambda wh: delta(wh[0], wh[1]))

def _squeeze(x):
    arr = np.asarray(x)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = arr.squeeze()
    return arr

def load_intrinsics(npz_path, k_key=None, d_key=None):
    data = np.load(npz_path, allow_pickle=True)
    K=None; D=None
    if k_key:
        if k_key not in data: raise RuntimeError(f"--k_key '{k_key}' not found. Keys: {list(data.files)}")
        K = _squeeze(data[k_key]).astype(np.float64)
    if d_key:
        if d_key not in data: raise RuntimeError(f"--d_key '{d_key}' not found. Keys: {list(data.files)}")
        D = _squeeze(data[d_key]).astype(np.float64)
    if K is None:
        K = _first_present(data, ["K","camera_matrix","mtx","K0"])
        if K is not None: K = _squeeze(K).astype(np.float64)
    if D is None:
        D = _first_present(data, ["D","dist","distCoeffs","D0"])
        if D is not None: D = _squeeze(D).astype(np.float64)
    if (K is None or K.shape!=(3,3)) or (D is None or not isinstance(D, np.ndarray)):
        # heuristics
        for k in data.files:
            v = _squeeze(data[k])
            if isinstance(v, np.ndarray) and v.shape==(3,3) and K is None:
                K=v.astype(np.float64); print(f"[INFO] Heuristic: using '{k}' as K")
            if isinstance(v, np.ndarray) and v.ndim==1 and v.size in (4,5,8,12,14,16) and D is None:
                D=v.astype(np.float64); print(f"[INFO] Heuristic: using '{k}' as D")
    size_raw = _first_present(data, ["image_size","size","wh","resolution","img_size"])
    size = _to_size_tuple(size_raw)
    if K is None or K.shape!=(3,3) or D is None:
        raise RuntimeError(f"Intrinsics NPZ {npz_path} missing usable K(3x3) and/or D(vector). Keys: {list(data.files)}")
    return K, D, size

def scale_K_if_needed(K, saved_size, img_shape_wh):
    K = K.copy()
    if saved_size is None: return K
    saved_w_h = _pick_size_orientation(saved_size, img_shape_wh)
    w_saved, h_saved = int(saved_w_h[0]), int(saved_w_h[1])
    w_img, h_img = int(img_shape_wh[0]), int(img_shape_wh[1])
    if (w_saved, h_saved) != (w_img, h_img):
        sx, sy = w_img/float(w_saved), h_img/float(h_saved)
        K[0,0] *= sx; K[1,1] *= sy
        K[0,2] *= sx; K[1,2] *= sy
        print(f"[INFO] Scaled K: saved=({w_saved},{h_saved}) -> img=({w_img},{h_img}); sx={sx:.6f}, sy={sy:.6f}")
    return K

def chessboard_object_points(rows, cols, square):
    obj = np.zeros((rows*cols, 3), np.float64)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:,0] = xs.ravel() * square
    obj[:,1] = ys.ravel() * square
    return obj

def detect_chessboard(img, rows, cols, subpixel=True):
    gray = img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret=False; corners=None
    pattern_size=(cols,rows)
    try:
        ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    except Exception: pass
    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if ret and subpixel and corners is not None:
            term=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
            cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term)
    if isinstance(corners, np.ndarray):
        corners=corners.reshape(-1,1,2).astype(np.float64)
    return bool(ret), corners

def build_charuco_board(squaresX, squaresY, square, marker=None, dict_name="DICT_5X5_1000"):
    if not hasattr(cv2,"aruco"): raise RuntimeError("OpenCV aruco module not available.")
    ar=cv2.aruco
    dictionary=getattr(ar, dict_name)
    if isinstance(dictionary,int): dictionary=ar.getPredefinedDictionary(dictionary)
    marker_len=float(square)*0.75 if marker is None else float(marker)
    board=ar.CharucoBoard((int(squaresX),int(squaresY)), float(square), marker_len, dictionary)
    return board, dictionary

def detect_charuco(img, board, dictionary, subpixel=True):
    ar=cv2.aruco
    gray=img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector=ar.ArucoDetector(dictionary, ar.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids)==0: return False, None, None
    if subpixel:
        for c in corners:
            cv2.cornerSubPix(gray, c, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
    retval, ch_corners, ch_ids = ar.interpolateCornersCharuco(corners, ids, gray, board)
    if retval is None or ch_corners is None or ch_ids is None or len(ch_ids)<4:
        return False, None, None
    return True, ch_corners.astype(np.float64), ch_ids.flatten()

def solve_pose_and_rms(obj_pts, img_pts, K, D, fisheye=False):
    img_pts=np.asarray(img_pts,np.float64).reshape(-1,1,2)
    obj_pts=np.asarray(obj_pts,np.float64).reshape(-1,3)
    if obj_pts.shape[0] < 4: return None, None, None, np.inf, None, None
    if fisheye:
        und=cv2.fisheye.undistortPoints(img_pts,K,D,P=K)
    else:
        und=cv2.undistortPoints(img_pts,K,D,P=K)
    und_pts=und.reshape(-1,1,2)
    ok,rvec,tvec,inl=cv2.solvePnPRansac(obj_pts,und_pts,K,None,iterationsCount=1000,reprojectionError=1.2,confidence=0.999,flags=cv2.SOLVEPNP_AP3P)
    if not ok or rvec is None:
        ok2,rvec2,tvec2=cv2.solvePnP(obj_pts,und_pts,K,None,flags=cv2.SOLVEPNP_EPNP)
        if not ok2: return None, None, None, np.inf, None, None
        rvec,tvec=rvec2,tvec2
    if isinstance(inl,np.ndarray) and inl.size>=max(6,int(0.6*len(obj_pts))):
        mask=np.zeros(len(obj_pts),bool); mask[inl.reshape(-1)]=True
        obj_use, und_use = obj_pts[mask], und_pts[mask]
        img_use = img_pts[mask]
        inlier_mask = mask
    else:
        obj_use, und_use = obj_pts, und_pts
        img_use = img_pts
        inlier_mask = np.ones(len(obj_pts), bool)
    rvec, tvec = cv2.solvePnPRefineLM(obj_use, und_use, K, None, rvec, tvec)
    if fisheye:
        reproj,_=cv2.fisheye.projectPoints(obj_use,rvec,tvec,K,D)
    else:
        reproj,_=cv2.projectPoints(obj_use,rvec,tvec,K,D)
    reproj=reproj.reshape(-1,2)
    err=np.sqrt(np.sum((reproj - img_use.reshape(-1,2))**2, axis=1))
    rms=float(np.sqrt(np.mean(err**2)))
    return rvec,tvec,err,rms,reproj,inlier_mask

def draw_overlay(img, img_pts, reproj_pts, title=None):
    vis = img.copy() if img.ndim==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (u,v) in img_pts.reshape(-1,2):
        cv2.circle(vis,(int(round(u)),int(round(v))),5,(0,255,0),2,cv2.LINE_AA)
    for (u,v) in reproj_pts.reshape(-1,2):
        cv2.drawMarker(vis,(int(round(u)),int(round(v))),(0,0,255),markerType=cv2.MARKER_TILTED_CROSS,markerSize=14,thickness=2)
    if title:
        cv2.putText(vis,title,(18,36),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2,cv2.LINE_AA)
    return vis

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--intrinsics", required=False)
    ap.add_argument("--npz_list", action="store_true")
    ap.add_argument("--k_key"); ap.add_argument("--d_key")
    g=ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--glob", action="append")
    g.add_argument("--images", nargs="+")
    ap.add_argument("--mode", choices=["chessboard","charuco"], default="chessboard")
    ap.add_argument("--rows", type=int); ap.add_argument("--cols", type=int)
    ap.add_argument("--squaresX", type=int); ap.add_argument("--squaresY", type=int)
    ap.add_argument("--dict", default="DICT_5X5_1000")
    ap.add_argument("--marker", type=float)
    ap.add_argument("--square", type=float, help="Square side length (units arbitrary but consistent)")
    ap.add_argument("--fisheye", action="store_true")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--out_csv")
    ap.add_argument("--dump_residuals", help="Write per-point residuals CSV (one row per detected corner)")
    ap.add_argument("--overlay_dir")
    args=ap.parse_args()
    # =========================
    #       APPLY PRESET
    # =========================
    args.glob = PRESET["glob"]
    args.intrinsics = PRESET["intrinsics"]
    args.rows = PRESET["rows"]
    args.cols = PRESET["cols"]
    args.square = PRESET["square"]
    args.save_overlay = PRESET["save_overlay"]
    args.overlay_dir = PRESET["overlay_dir"]

    if args.npz_list: return npz_list_keys(args.intrinsics)
    if args.glob is None and args.images is None:
        print("[ERR] Provide --glob or --images, or use --npz_list"); return 2

    paths=[]
    if args.images: paths.extend(args.images)
    else:
        for pat in args.glob: paths.extend(glob.glob(pat))
    paths=[p for p in paths if os.path.isfile(p)]
    if not paths: print("[ERR] No images found."); return 2
    paths.sort()

    K_raw,D,saved_size = load_intrinsics(args.intrinsics, args.k_key, args.d_key)

    if args.mode=="chessboard":
        if args.rows is None or args.cols is None or args.square is None:
            print("[ERR] --rows --cols --square required for chessboard"); return 2
        obj_full = chessboard_object_points(args.rows, args.cols, float(args.square))
    else:
        if args.squaresX is None or args.squaresY is None or args.square is None:
            print("[ERR] --squaresX --squaresY --square required for charuco"); return 2
        board, dictionary = build_charuco_board(args.squaresX, args.squaresY, float(args.square), args.marker, args.dict)

    rows_csv=[]
    if args.out_csv: rows_csv.append(["image","n_pts","rms_px","median_px","max_px"])
    if args.dump_residuals:
        res_header=["image","idx","u","v","uhat","vhat","du","dv","r_px","theta_deg","inlier","cx","cy","fx","fy","rms_img"]
        res_rows=[]

    all_rms=[]
    print(f"[INFO] Auditing {len(paths)} image(s)...\n")
    for i,pth in enumerate(paths,1):
        img=cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if img is None: print(f"[WARN] Could not read {pth}"); continue
        Himg,Wimg = img.shape[:2]
        K = scale_K_if_needed(K_raw, saved_size, (Wimg,Himg))
        cx,cy,fx,fy = K[0,2], K[1,2], K[0,0], K[1,1]

        if args.mode=="chessboard":
            ok,corners = detect_chessboard(img, args.rows, args.cols, subpixel=True)
            if not ok: print(f"[WARN] Chessboard not found: {pth}"); continue
            obj=obj_full.copy(); img_pts=corners
        else:
            ok,ch_corners,ch_ids = detect_charuco(img, board, dictionary, subpixel=True)
            if not ok: print(f"[WARN] ChArUco not found: {pth}"); continue
            obj = board.chessboardCorners[ch_ids].reshape(-1,3).astype(np.float64)
            img_pts = ch_corners.reshape(-1,1,2).astype(np.float64)

        rvec,tvec,err,rms,reproj,inlier_mask = solve_pose_and_rms(obj, img_pts, K, D, fisheye=args.fisheye)
        if rms==np.inf or rvec is None: print(f"[WARN] Pose solve failed: {pth}"); continue

        med=float(np.median(err)) if err is not None and len(err)>0 else float("nan")
        mx=float(np.max(err)) if err is not None and len(err)>0 else float("nan")
        all_rms.append(rms)

        print(f"[{i:03d}/{len(paths)}] {os.path.basename(pth)}  n={len(obj):3d}  RMS={rms:6.3f} px  median={med:6.3f}  max={mx:6.3f}")

        if args.save_overlay:
            vis = draw_overlay(img, img_pts[inlier_mask], reproj, title=f"RMS={rms:.2f}px  n={len(reproj)}")
            os.makedirs(args.overlay_dir, exist_ok=True)

            out_png = os.path.join(
                args.overlay_dir,
                os.path.splitext(os.path.basename(pth))[0] + "_audit.png"
            )

            cv2.imwrite(out_png, vis)

        if args.out_csv:
            rows_csv.append([pth, len(obj), f"{rms:.6f}", f"{med:.6f}", f"{mx:.6f}"])

        if args.dump_residuals:
            img_flat = img_pts[inlier_mask].reshape(-1,2)
            for j, ((u,v),(uh,vh),e) in enumerate(zip(img_flat, reproj, err)):
                du, dv = (uh-u), (vh-v)
                dx, dy = (u-cx), (v-cy)
                r = float(np.sqrt(dx*dx + dy*dy))
                theta = float(np.degrees(np.arctan2(dy, dx)))
                res_rows.append([pth, j, u, v, uh, vh, du, dv, r, theta, 1, cx, cy, fx, fy, rms])

    if not all_rms:
        print("\n[RESULT] No successful detections/poses. Check your pattern settings or images.")
        return 1

    if args.out_csv:
        with open(args.out_csv,"w",newline="") as f:
            w=csv.writer(f); w.writerows(rows_csv)
        print(f"[INFO] Wrote per-image CSV: {args.out_csv}")

    if args.dump_residuals and len(res_rows)>0:
        with open(args.dump_residuals,"w",newline="") as f:
            w=csv.writer(f); w.writerow(res_header); w.writerows(res_rows)
        print(f"[INFO] Wrote residuals CSV: {args.dump_residuals}  (rows={len(res_rows)})")

    print("\n[RESULT] Intrinsics audit summary")
    all_rms=np.array(all_rms,float)
    print(f"  Images used    : {len(all_rms)}")
    print(f"  Median RMS     : {np.median(all_rms):.3f} px")
    print(f"  Mean RMS       : {np.mean(all_rms):.3f} px")
    print(f"  90th percentile: {np.percentile(all_rms,90):.3f} px")
    print(f"  Max RMS        : {np.max(all_rms):.3f} px")
    return 0

if __name__=="__main__":
    sys.exit(main())
