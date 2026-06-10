import cv2
import os
import time
from cam_stream import CameraStream

# =========================
# CAMERA SELECTION
# =========================

CAM = "left"   # change to "left", "right", or "back"            #********************* change this when you switch cameras

# Configuration
DEVICE_INDEX = 2                          # Camera device index
OUTPUT_DIR = f"./data/{CAM}_captures"     # Directory where frames will be saved
INTERVAL = 1.5                            # Seconds between captures
DISPLAY_SCALE = 0.25                      # Scale for real-time display (1/4 area)


def capture_images(device_index=DEVICE_INDEX, output_dir=OUTPUT_DIR, interval=INTERVAL):
    """
    Capture an image from the specified camera once per `interval` seconds,
    saving to `output_dir`, until 'q' is pressed in the display window.
    The live view is shown at a fraction of full resolution defined by DISPLAY_SCALE.
    """
    # Initialize the threaded camera stream
    stream = CameraStream(device_index=device_index)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print("Press 'q' in the image window to stop capturing.")

    count = 0
    try:
        while True:
            frame = stream.get_latest_frame()
            if frame is None:
                print("Warning: No frame received yet")
                time.sleep(0.1)
                continue

            # Resize for display only
            disp_frame = cv2.resize(
                frame,
                (0, 0),
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_AREA
            )

            # Display the scaled frame
            cv2.imshow('Live Capture - Press q to Quit', disp_frame)

            # Save the full-resolution frame with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(
                output_dir,
                f'cam{device_index}_{timestamp}_{count:03d}.png'
            )
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1

            # Wait for the interval or until 'q' is pressed
            start = time.time()
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping capture.")
                    return
                if time.time() - start >= interval:
                    break
    finally:
        # Cleanup
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_images()
