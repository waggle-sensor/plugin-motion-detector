from threading import Thread
from queue import Queue
import cv2

def create_video_capture_queue(device):
    frames = Queue(30)

    def video_capture_worker():
        capture = cv2.VideoCapture(device)
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to capture frame")
            frames.put(frame)

    Thread(target=video_capture_worker, daemon=True).start()
    return frames
