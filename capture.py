from threading import Thread
from queue import Queue
import cv2

def create_video_capture_queue(device, queue_size=30):
    frames = Queue(queue_size)

    def video_capture_worker():
        capture = cv2.VideoCapture(device)
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to capture frame")
            frames.put(frame)

    Thread(target=video_capture_worker, daemon=True).start()
    return frames
