import cv2
import threading
import time

class CameraStream:
    def __init__(self, device_index=0, width=5472, height=3648, fps=4, focus_value=None):
        # Open capture
        self.cap = cv2.VideoCapture(device_index)

        # Disable autofocus if supported
#        if self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) is False:
#            print("⚠️ Warning: Could not disable autofocus on camera {}".format(device_index))
#        # Optionally set manual focus
#        if focus_value is not None:
#            if self.cap.set(cv2.CAP_PROP_FOCUS, focus_value) is False:
#                print(f"⚠️ Warning: Could not set manual focus to {focus_value}")

        # Set resolution and frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Internal state
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()


    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.2)  # Reduce CPU usage and USB saturation

    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
