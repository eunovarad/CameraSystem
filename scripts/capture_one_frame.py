import cv2
import os

# =========================
# SIMPLE ONE-FRAME CAPTURE
# =========================

# LAPTOP TERMINAL COMMAND:
# C:\Users\EUlibarri\AppData\Local\Python\pythoncore-3.14-64\python.exe scripts\capture_one_frame.py

CAM = "left"           # change to "left", "right", or "back"
SET_NAME = "set01"     # use the same set name for all 3 cameras
DEVICE_INDEX = 0       # vary as needed

OUTPUT_DIR = "./data/phantom_images"

CAM_WIDTH = 5472
CAM_HEIGHT = 3648
CAM_FPS = 5

USE_DSHOW = True       # recommended on Windows
USE_MJPG = True        # often helps USB camera reliability


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_DSHOW:
        cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(DEVICE_INDEX)

    if not cap.isOpened():
        print(f"❌ Could not open camera at index {DEVICE_INDEX}")
        return

    # Optional reliability settings
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    filename = f"{CAM}_{SET_NAME}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)

    print("Live preview started.")
    print("Press SPACE to save one image and exit.")
    print("Press Q to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ No frame received yet")
            continue

        display = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cv2.imshow("One-Frame Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exited without saving.")
            break

        if key == 32:  # SPACE
            cv2.imwrite(out_path, frame)
            print(f"✅ Saved: {out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
