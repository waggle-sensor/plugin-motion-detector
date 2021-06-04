#!/bin/env python3

import argparse
import time
import logging
import cv2
from capture import create_video_capture_queue
from detectors import EMAObjectDetector, BGSubObjectDetector,DenseOpticalFlowDetector

def main():
    parser = argparse.ArgumentParser(description="This program uses simple motion detection and background subtraction for object detection.")
    parser.add_argument("--input", default="/dev/video0", help="video input source")
    parser.add_argument("--display", action="store_true", help="display object detection preview")
    parser.add_argument("--filtered", action="store_true", help="display filtered input")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")

    logging.info("opencv version %s", cv2.__version__)

    #detector = BGSubObjectDetector(cv2.createBackgroundSubtractorMOG2())
    #detector = BGSubObjectDetector(cv2.createBackgroundSubtractorKNN())
    #detector = EMAObjectDetector(0.5)
    detector = DenseOpticalFlowDetector()

    frames = create_video_capture_queue(args.input)

    try:
        while True:
            logging.info("getting frame")
            frame = frames.get(timeout=10.0)

            logging.info("applying detector")
            objects = detector.apply(frame)
            if args.filtered and (detector.filtered_frame is not None):
                frame = detector.filtered_frame
                #frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            
            for x, y, w, h in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                logging.info("object in x=%s y=%s w=%s h=%s", x, y, w, h)

            if args.display:
                cv2.imshow("Preview", frame)
                keyboard = cv2.waitKey(1) & 0xFF
                if keyboard == ord("q") or keyboard == 27:
                    break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
