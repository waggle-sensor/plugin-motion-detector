from threading import Thread
from queue import Queue
import time
import cv2

def create_video_capture_queue(device, queue_size=30, fps=None):
    frames = Queue(queue_size)
    

    def video_capture_worker():
        capture = cv2.VideoCapture(device)
        next_cap = time.time()
        current_time = next_cap
        while capture.isOpened():
            current_time = time.time()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to capture frame")
            elif fps is None: 
                frames.put(frame)
            elif current_time >= next_cap:
                frames.put(frame)
                next_cap = current_time + 1./fps

    Thread(target=video_capture_worker, daemon=True).start()
    return frames
