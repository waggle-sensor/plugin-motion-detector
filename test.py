from unittest import main, TestCase
import cv2
import time

from capture import create_video_capture_queue
from detector import EMAObjectDetector, BGSubObjectDetector,DenseOpticalFlowDetector, YOLODetector
from object_tracker import TrackedObjectDatabase, EMATracker

# This is a sequence of images displaying a bouncing ball:
TEST_OBJECT_FRAMES = './test/ball-%2d.png' 

# These are the locations of the bouncing ball center:
TEST_OBJECT_LOCATIONS= [
    (222, 70),
    (222, 70),
    (222, 70),
    (222, 70),
    (222, 70),
    (222, 77),
    (222, 98),
    (222, 198),
    (222, 259),
    (222, 264),
    (222, 267),
    (222, 259),
    (222, 243),
    (222, 215),
    (222, 186),
    (221, 157),
    (222, 129),
    (222, 100),
    (222, 71),
    (222, 45)        
]

def _pt_within_rect(pt, rect, margin=50):
    return  (rect[0]-margin <= pt[0] <= rect[0]+rect[2]+margin) \
        and (rect[1]-margin <= pt[1] <= rect[1]+rect[3]+margin)


def _test_detector(detector):
    """Tests a detector on an animated image"""
    frames = create_video_capture_queue(TEST_OBJECT_FRAMES, quiet=True)
    for i, pt in enumerate(TEST_OBJECT_LOCATIONS): 
        frame = frames.get(timeout=1.0)
        rects = detector.apply(frame)
        if i > 1:
            assert len(rects) == 1, \
                f"exactly one object must be discovered (found {len(rects)})"
            
            for rect in rects:
                (x,y,w,h) = rect
                assert _pt_within_rect(pt,rect), \
                    f"object center must be within discovered rectangle (pt: {pt}, rect: ({x},{y};w={w},h={h}))" 

def _test_tod(detector, tracker):
    """Tests a tracked object database on an animated image"""
    tod = TrackedObjectDatabase(detector, tracker)
    frames = create_video_capture_queue(TEST_OBJECT_FRAMES, quiet=True)
    for i, pt in enumerate(TEST_OBJECT_LOCATIONS): 
        frame = frames.get(timeout=1.0)
        tod.update_tracked_objects(frame)
        objs = tod.get_tracked_objects_info(with_meta=False)
        if i > 1:
            for  obj in objs.values():
                assert 'label' in obj
                assert 'rect' in obj
                assert 'last_seen' in obj
                assert obj['last_seen'] <= time.time(), "time object was last seen must be before now"

class BGSubDetectorTest(TestCase):
        
    def test_detector(self):
        detector = BGSubObjectDetector(1,cv2.createBackgroundSubtractorKNN())
        _test_detector(detector)
    
    def test_tod(self):
        detector = BGSubObjectDetector(1,cv2.createBackgroundSubtractorKNN())
        tracker = EMATracker(weight=1.0, object_ttl=1.0)
        _test_tod(detector, tracker)

class DenseOpticalFlowDetectorTest(TestCase):
    
    def test_detector(self):
        detector = DenseOpticalFlowDetector(1) 
        _test_detector(detector)

    def test_tod(self):
        detector = DenseOpticalFlowDetector(1) 
        tracker = EMATracker(weight=1.0, object_ttl=1.0)
        _test_tod(detector, tracker)

class YoloDetectorTest(TestCase):

    def test_detector(self):
        pass

    def test_tod(self):
        detector = YOLODetector()
        tracker = EMATracker(weight=1.0, object_ttl=1.0)
        _test_tod(detector, tracker)

if __name__ == '__main__':
    main()
