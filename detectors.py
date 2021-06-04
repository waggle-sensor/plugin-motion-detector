import cv2
import numpy as np

# Detectors should satisfy the interface:
#   detector.apply(frame) -> object list [ returns list of moving objects      ]
#   detector.reset()                     [ resets the detector for a new scene ]

class EMAObjectDetector:
    """
        Detects objects via an exponential moving average in grayscale
        color difference

    """
    def __init__(self, weight):
        self.avg = None
        self.weight = weight
        self.filtered_frame = None


    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (256, 256), 0)
        if self.avg is None:
            self.avg = gray
        else:
            self.avg = (self.weight * self.avg + (1-self.weight) * gray).astype(np.uint8)
        delta = cv2.absdiff(gray, self.avg)
        _, thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.filtered_frame = thresh
        return get_bounding_boxes_from_thresh(thresh, min_area=2000)
    
    def reset(self):
        self.avg = None


class BGSubObjectDetector:
    """
        Detects objects via Background subtraction
    """

    def __init__(self, bgsub):
        self.bgsub = bgsub
        self.filtered_frame = None
    
    def apply(self, frame):
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fg = self.bgsub.apply(gray)
        _, thresh = cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.filtered_frame = thresh
        return get_bounding_boxes_from_thresh(thresh, min_area=2000)

    def reset(self):
        pass

class DenseOpticalFlowDetector:
    """
        Detects moving objects through optical flow
    """
    
    def __init__(self):
        self.lastgray = None
        self.filtered_frame = None

    def apply(self, frame):
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (41, 41), 0)
        if self.lastgray is None:
            self.lastgray = gray
        flow = cv2.calcOpticalFlowFarneback(self.lastgray, gray, None,
                                            0.2, 1, 12, 2, 2, 1.2, 0) 
        r, theta = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(frame)
        hsv[...,0] = theta * 180 / np.pi / 2
        hsv[...,1] = 255
        hsv[...,2] = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        self.lastgray = gray
        flow_img = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),(0,0),fx=2.0, fy=2.0)
        self.filtered_frame = flow_img
        
        gray_flow = cv2.cvtColor(flow_img,cv2.COLOR_BGR2GRAY)
        _, flow_thresh = cv2.threshold(gray_flow, 100, 255, cv2.THRESH_BINARY)
        flow_thresh= cv2.dilate(flow_thresh, None, iterations=2)
        return get_bounding_boxes_from_thresh(flow_thresh, min_area=2000)

    def reset(self):
        self.lastgray = None


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
