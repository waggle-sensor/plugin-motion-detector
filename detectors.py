import cv2
import numpy as np

# Detectors should satisfy the interface:
# detector.apply(frame) -> object list

class EMAObjectDetector:

    def __init__(self, weight):
        self.avg = None
        self.weight = weight
    
    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.avg is None:
            self.avg = gray
        else:
            self.avg = (self.weight * self.avg + (1-self.weight) * gray).astype(np.uint8)
        delta = cv2.absdiff(gray, self.avg)
        _, thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        return get_bounding_boxes_from_thresh(thresh, min_area=2000)


class BGSubObjectDetector:

    def __init__(self, bgsub):
        self.bgsub = bgsub
    
    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fg = self.bgsub.apply(gray)
        _, thresh = cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        return get_bounding_boxes_from_thresh(thresh, min_area=2000)


def get_bounding_boxes_from_thresh(thresh, min_area):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_area]

# TODO can we bootstrap / parameter search for a lightweight motion tracker
# via "transfer learning" of a powerful object detector? or at least generate
# data for validation (like iou rects
# TODO generate test frames with moving rect. we can *place* the rect and then
# estimate the detection (iou metrics n such).
# TODO find existing traffic / object dataset
# TODO test blender render to generate bboxes
