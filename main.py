import argparse
import time
import logging
import cv2
from capture import create_video_capture_queue
from detectors import EMAObjectDetector, BGSubObjectDetector

def main():
    parser = argparse.ArgumentParser(description="This program uses simple motion detection and background subtraction for object detection.")
    parser.add_argument("--display", action="store_true", help="display object detection preview")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    logging.info("opencv version %s", cv2.__version__)

    # detector = BGSubObjectDetector(cv2.createBackgroundSubtractorMOG2())
    # detector = BGSubObjectDetector(cv2.createBackgroundSubtractorKNN())
    detector = EMAObjectDetector(0.7)

    frames = create_video_capture_queue("test.mp4")

    try:
        while True:
            frame = frames.get()
            objects = detector.apply(frame)
            for x, y, w, h in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if args.display:
                cv2.imshow("Preview", frame)
                keyboard = cv2.waitKey(1) & 0xFF
                if keyboard == ord("q") or keyboard == 27:
                    break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
