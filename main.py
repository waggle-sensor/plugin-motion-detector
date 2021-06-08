#!/bin/env python3

import argparse
import time
import logging
import cv2
from capture import create_video_capture_queue
from detector import EMAObjectDetector, BGSubObjectDetector,DenseOpticalFlowDetector
from object_tracker import TrackedObjectDatabase, EMATracker

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
    detector = DenseOpticalFlowDetector(4)

    tracker = EMATracker()

    frames = create_video_capture_queue(args.input)
    tod = TrackedObjectDatabase(detector, tracker)

    try:
        while True:
            #logging.info("getting frame")
            frame = frames.get(timeout=10.0)

            #logging.info("applying tracker")
            tod.update_tracked_objects(frame)
            if args.filtered and (tod.detector.filtered_frame is not None):
                frame = tod.detector.filtered_frame
        
            tod.show_tracked_objects(frame)

            if args.display:
                cv2.imshow("Preview (press \'q\' to quit)", frame)
                keyboard = cv2.waitKey(1) & 0xFF
                if keyboard == ord("q") or keyboard == 27:
                    break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
