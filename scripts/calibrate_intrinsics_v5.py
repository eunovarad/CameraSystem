#!/usr/bin/env python3
"""
calibrate_intrinsics_v5.py

==============================================
SIMPLIFIED USAGE (RECOMMENDED FOR NORMAL USE)
==============================================

This script supports MANY advanced options, but most of the time you only need:
→ Standard checkerboard calibration
To make this easy, we added a "PRESET MODE".

----------------------------------------------
HOW TO USE PRESET MODE:
----------------------------------------------
Set:
    USE_PRESET = True

Then edit PRESET values below if needed.
----------------------------------------------
WHEN TO TURN PRESets OFF:
----------------------------------------------
Set:
    USE_PRESET = False
Then you can use full CLI arguments again.
----------------------------------------------
"""

# =========================
# CAMERA SELECTION
# =========================

CAM = "left"   # change to "left", "right", or "back"            #********************* change this when you switch cameras

import argparse, glob, os, sys, csv, time, traceback
from datetime import datetime
import numpy as np
import cv2


# =========================
#   SIMPLE PRESET CONFIG
# =========================

USE_PRESET = True

PRESET = {
    "glob": [f"./data/{CAM}_captures/*.png"],
    "mode": "chessboard",
    "rows": 7,
    "cols": 11,
    "square": 20.0,
    "solver": "classic",
    "calib_scale": 1.0,
    "save_npz": f"intrin_cam_{CAM}.npz",
    "save_overlay": True,
    "overlay_dir": f"./data/{CAM}_overlays",
    "verbose": True
}

def say(m=""): print(m, flush=True)
def banner(m): say("="*64); say(m); say("="*64)


# ---------------- helpers ----------------

def chessboard_object_points(rows, cols, square):
    xs, ys = np.meshgrid(np.arange(cols, dtype=np.float32),
                         np.arange(rows, dtype=np.float32))
    obj = np.zeros((rows*cols, 1, 3), np.float32)
    obj[:,0,0] = xs.ravel() * float(square)
    obj[:,0,1] = ys.ravel() * float(square)
    return obj


def detect_chessboard(img, rows, cols):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pattern = (cols, rows)

    ok, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )

    if ok:
        corners = corners.astype(np.float32)

    return ok, corners



# ---------------- main ----------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--glob", action="append")
    ap.add_argument("--mode", choices=["chessboard"], default="chessboard")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--square", type=float)
    ap.add_argument("--solver", default="classic")
    ap.add_argument("--calib_scale", type=float, default=1.0)
    ap.add_argument("--save_npz", default="intrinsics.npz")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--overlay_dir", default="calib_overlays")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()


    # =========================
    # ✅ APPLY PRESET
    # =========================
    if USE_PRESET:
        say("[INFO] Using PRESET configuration")

        args.glob = PRESET["glob"]
        args.mode = PRESET["mode"]
        args.rows = PRESET["rows"]
        args.cols = PRESET["cols"]
        args.square = PRESET["square"]
        args.solver = PRESET["solver"]
        args.calib_scale = PRESET["calib_scale"]
        args.save_npz = PRESET["save_npz"]
        args.save_overlay = PRESET["save_overlay"]
        args.overlay_dir = PRESET["overlay_dir"]
        args.verbose = PRESET["verbose"]


    banner("[START CALIBRATION]")

    # Collect images
    paths = []
    for pat in args.glob:
        say(f"[GLOB] {pat}")
        m = glob.glob(pat)
        paths.extend(m)

    if not paths:
        say("[ERR] No images found")
        return

    paths.sort()
    say(f"[INFO] Found {len(paths)} images")

    objpoints=[]
    imgpoints=[]

    for i,p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue

        ok, corners = detect_chessboard(img, args.rows, args.cols)

        if ok:
            imgpoints.append(corners)
            objpoints.append(
                chessboard_object_points(args.rows, args.cols, args.square)
            )
            say(f"[{i}] OK ({len(corners)})")
        else:
            say(f"[{i}] FAIL")

    if len(objpoints) == 0:
        say("[ERR] No valid detections")
        return

    image_size = (img.shape[1], img.shape[0])

    say(f"[INFO] Calibrating using {len(objpoints)} valid images")

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    say(f"[RESULT] RMS: {rms:.4f}")

    np.savez(args.save_npz, K=K, D=D, image_size=image_size)

    say(f"[OK] Saved: {args.save_npz}")

    # overlays
    if args.save_overlay:
        os.makedirs(args.overlay_dir, exist_ok=True)

        for i,p in enumerate(paths[:len(imgpoints)]):
            img = cv2.imread(p)
            if img is None:
                continue

            ok, rvec, tvec = cv2.solvePnP(
                objpoints[i], imgpoints[i], K, D
            )

            if not ok:
                continue

            proj, _ = cv2.projectPoints(
                objpoints[i], rvec, tvec, K, D
            )

            vis = img.copy()

            for pt in imgpoints[i].reshape(-1,2):
                cv2.circle(vis, tuple(pt.astype(int)), 5, (0,255,0), 2)

            for pt in proj.reshape(-1,2):
                cv2.drawMarker(vis, tuple(pt.astype(int)),
                    (0,0,255), cv2.MARKER_TILTED_CROSS, 10, 2)

            out = os.path.join(
                args.overlay_dir,
                os.path.basename(p).replace(".png","_overlay.png")
            )

            cv2.imwrite(out, vis)

    banner("[DONE]")


if __name__ == "__main__":
    main()