#!/bin/env python3

import argparse
import time
import logging
import cv2

import waggle.plugin as plugin
from waggle.data.vision import Camera, BGR

from detector import BGSubObjectDetector, DenseOpticalFlowDetector
# from detector YOLODetector
from object_tracker import TrackedObjectDatabase, EMATracker


def load_detector(name):
    """ Loads a detector by name """
    if name == 'bg_subtraction':
        return BGSubObjectDetector(1,cv2.createBackgroundSubtractorKNN())
    if name == 'dense_optflow':
        return DenseOpticalFlowDetector(1)
    # if name == 'yolo':
    #     return YOLODetector()
    raise Exception(f'Unknown detector: "{name}"')


def main():
    """ entrypoint for plugin """
    parser = argparse.ArgumentParser(description="This program uses simple motion detection and background subtraction for object detection.")
    parser.add_argument("--debug", action="store_true", help="enable debug logs")
    parser.add_argument("--input", default=0, help="video input source")
    parser.add_argument("--fps", type=float, default=None, help="frames per second of input source")
    parser.add_argument("--detector", default="bg_subtraction", help= \
    """
    The motion detector to use. In order from least to most computationally intensive, the options are:
        (1) bg_subtraction
        (2) dense_optflow
        (3) yolo
    """)
    parser.add_argument("--interval", type=float, default=10.0, help="interval between data publishes (in seconds)")
    parser.add_argument("--display", action="store_true", help="display object detection preview (for debugging)")
    parser.add_argument("--filtered", action="store_true", help="display filtered input (for debugging)")
    args = parser.parse_args()

    plugin.init()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S")
    logging.info("opencv version %s", cv2.__version__)

    camera = Camera(args.input, format=BGR)
    tod = TrackedObjectDatabase(load_detector(args.detector), EMATracker(object_ttl=1.0))
    publish_interval = args.interval
    next_publish = time.time()
    
    try:
        for sample in camera.stream():
            frame = sample.data
            tod.update_tracked_objects(frame)

            now = time.time()
            if now >= next_publish:
                # publish tracked object data:
                objs, objs_meta = tod.get_tracked_objects_info(with_meta=True)

                value = int(len(objs) > 0)
                plugin.publish('vision.motion_detected', value)
                logging.info('detected motion: %s', value)
                next_publish = now + publish_interval

            if args.display:
                if args.filtered and (tod.detector.filtered_frame is not None):
                    frame = tod.detector.filtered_frame
                tod.show_tracked_objects(frame)
                cv2.imshow("Preview (press \'q\' to quit)", frame)
                keyboard = cv2.waitKey(1) & 0xFF
                if keyboard == ord("q") or keyboard == 27:
                    break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
