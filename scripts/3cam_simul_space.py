#!/usr/bin/env python3
# 3cam_simul_cap_spacebar.py
# Multi-camera synchronized capture using CameraStream methodology.
# Capture is triggered by pressing the SPACE bar, not a timed interval.

import os
import time
import cv2
from cam_stream import CameraStream

# === HARD-CODED CONFIGURATION ===
CAM_IDS       = [0, 1, 2]                    # List of camera device indices
OUTPUT_DIR    = './data/phantom_images'      # Output directory for captured images
DISPLAY_SCALE = 0.20                         # Scale factor for display
CAM_NAMES = {
    0: "left",
    1: "right",
    2: "back"
}


# Match your single-cam capture settings:
CAM_WIDTH     = 5472
CAM_HEIGHT    = 3648
CAM_FPS       = 5
FOCUS_VALUE   = None                  # or an integer for manual focus

def open_streams(cam_ids):
    streams = {}
    for cid in cam_ids:
        streams[cid] = CameraStream(
            device_index=cid,
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            fps=CAM_FPS,
            focus_value=FOCUS_VALUE
        )
    return streams

def release_streams(streams):
    for s in streams.values():
        s.stop()
    cv2.destroyAllWindows()

def capture_loop(streams, cam_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    frames = {}
    print("Press SPACE to capture, 'q' to quit.")

    try:
        while True:
            # Grab & display latest frames
            for cid, stream in streams.items():
                frame = stream.get_latest_frame()
                if frame is None:
                    continue
                frames[cid] = frame
                disp = cv2.resize(
                    frame, (0, 0),
                    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA
                )
                cv2.imshow(f"Cam{cid}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == 32 and len(frames) == len(cam_ids):  # spacebar = 32
                ts = int(time.time() * 1000)
                for cid, frame in frames.items():
                    name = CAM_NAMES.get(cid, f"cam{cid}")
                    fname = f"{name}_{ts}.png"
                    path = os.path.join(OUTPUT_DIR, fname)
                    cv2.imwrite(path, frame)
                print(f"Captured set {count} @ {ts}")
                count += 1

    finally:
        release_streams(streams)

def main():
    streams = open_streams(CAM_IDS)
    capture_loop(streams, CAM_IDS, OUTPUT_DIR)

if __name__ == '__main__':
    main()
