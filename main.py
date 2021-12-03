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
    # parser.add_argument("--fps", type=float, default=20, help="frames per second of input source")
    parser.add_argument("--detector", default="dense_optflow", help= \
    """
    The motion detector to use. In order from least to most computationally intensive, the options are:
        (1) bg_subtraction
        (2) dense_optflow
        (3) yolo
    """)
    parser.add_argument("--samples", type=int, default=1, help="number of samples to publish")
    parser.add_argument("--interval", type=float, default=5.0, help="interval between data publishes (in seconds)")
    args = parser.parse_args()

    plugin.init()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S")
    logging.info("opencv version %s", cv2.__version__)

    camera = Camera(args.input, format=BGR)
    tod = TrackedObjectDatabase(load_detector(args.detector), EMATracker(object_ttl=1.0))
    publish_interval = args.interval
    next_publish = time.time() + publish_interval
    total_published = 0

    for sample in camera.stream():
        if args.samples > 0 and total_published >= args.samples:
            break

        frame = sample.data
        tod.update_tracked_objects(frame)

        now = time.time()
        if now < next_publish:
            continue

        # publish tracked object data:
        objs, _ = tod.get_tracked_objects_info(with_meta=True)

        value = int(len(objs) > 0)
        plugin.publish('vision.motion_detected', value)
        logging.info('detected motion: %s', value)
        next_publish = now + publish_interval
        total_published += 1

if __name__ == "__main__":
    main()
