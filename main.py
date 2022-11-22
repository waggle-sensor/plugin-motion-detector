#!/bin/env python3
import argparse
import time
import logging
import cv2
from contextlib import contextmanager

from pathlib import Path
from waggle.plugin import Plugin
from waggle.data.vision import Camera, BGR

from detector import BGSubObjectDetector, DenseOpticalFlowDetector
# from detector YOLODetector
from object_tracker import TrackedObjectDatabase, EMATracker


@contextmanager
def log_time(name):
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    logging.info(f'section {name} took {duration}s')


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

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S")
    logging.info(f'opencv version {cv2.__version__}')

    publish_interval = args.interval
    next_publish = time.time() + publish_interval
    total_published = 0

    with log_time("setup"):
        tod = TrackedObjectDatabase(load_detector(args.detector), EMATracker(object_ttl=1.0))

    with Plugin() as plugin, Camera(args.input, format=BGR) as camera:
        for sample in camera.stream():
            if args.samples > 0 and total_published >= args.samples:
                break

            frame = sample.data
            with log_time("update"):
                tod.update_tracked_objects(frame)

            now = time.time()
            if now < next_publish:
                continue

            # publish tracked object data:
            with log_time("publish"):
                objs, _ = tod.get_tracked_objects_info(with_meta=True)
                logging.info(objs)
                value = int(len(objs) > 0)
                plugin.publish('vision.motion_detected', value)
                logging.info(f'vision.motion_detected: {value}')

            logging.info(f'detected motion: {value}')
            next_publish = now + publish_interval
            total_published += 1

            if tod.detector.filtered_frame is not None:
                frame = tod.detector.filtered_frame
            tod.show_tracked_objects(frame)
            cv2.imwrite('result.jpg', frame)
            plugin.upload_file('result.jpg')
            logging.info("A result is published")

if __name__ == "__main__":
    main()
