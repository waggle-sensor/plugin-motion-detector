import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import tensornets as nets

# Detectors should satisfy the interface:
#   detector.apply(frame) -> object list [ returns list of moving objects      ]
#   detector.reset()                     [ resets the detector for a new scene ]
#
# Optionally, detectors may also have an attribute:
#       <detector>.filtered_frame
#  which should be updated within a call to the apply() function. This is set to some
#  filtered representation of the frame, and can be shown (instead of the raw camera feed)
#  when both the --display and --filtered flags are used.

class EMAObjectDetector:
    """Detects moving objects via an exponential moving average of color differences"""
    def __init__(self, max_n_objs, weight):
        self.avg = None
        self.max_n_objs = max_n_objs
        self.weight = weight
        self.filtered_frame = None


    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (41, 41), 0)
        if self.avg is None:
            self.avg = gray
        else:
            self.avg = (self.weight * self.avg + (1-self.weight) * gray).astype(np.uint8)
        delta = cv2.absdiff(gray, self.avg)
        _, thresh = cv2.threshold(delta, 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.filtered_frame = thresh
        return get_bounding_boxes_from_thresh(thresh, 2000, self.max_n_objs)
    
    def reset(self):
        self.avg = None


class BGSubObjectDetector:
    """Detects moving objects via background subtraction"""

    def __init__(self, max_n_objs, bgsub):
        self.bgsub = bgsub
        self.max_n_objs = max_n_objs
        self.filtered_frame = None
    
    def apply(self, frame):
   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fg = self.bgsub.apply(gray)
        _, thresh = cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.filtered_frame = thresh
        return get_bounding_boxes_from_thresh(thresh, 2000, self.max_n_objs)

    def reset(self):
        pass

class DenseOpticalFlowDetector:
    """Detects moving objects through optical flow"""
    
    def __init__(self, max_n_objs=4, r_mean=0.5, r_stddev=0.3, r_thresh=0.2):
        self.lastgray = None
        self.max_n_objs = max_n_objs
        self.r_mean = r_mean
        self.r_stddev = r_stddev
        self.r_thresh = r_thresh
        self.filtered_frame = None

    def apply(self, frame):
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.lastgray is None:
            self.lastgray = gray
        flow = cv2.calcOpticalFlowFarneback(self.lastgray, gray, None,
                                            0.5, # pyramid scale
                                            3,   # levels 
                                            32,  # window size
                                            3,   # iterations
                                            5,   # polynomial degree
                                            1.2, # polynomial std. dev.
                                            0) 
        
        r, theta = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(frame)
        hsv[...,0] = theta * 180 / np.pi / 2
        hsv[...,1] = 255
        hsv[...,2] = np.clip(255*(r-self.r_mean)/self.r_stddev, 0,255) 
        self.lastgray = gray
        flow_img = cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),(0,0),fx=2.0, fy=2.0)

        gray_flow = cv2.cvtColor(flow_img,cv2.COLOR_BGR2GRAY)
        self.filtered_frame = flow_img
        _, flow_thresh = cv2.threshold(gray_flow, 255*self.r_thresh, 255, cv2.THRESH_BINARY)
        flow_thresh= cv2.dilate(flow_thresh, None, iterations=4)
        #self.filtered_frame = flow_thresh
        return get_bounding_boxes_from_thresh(flow_thresh, 2000, self.max_n_objs)

    def reset(self):
        self.lastgray = None

def get_bounding_boxes_from_thresh(thresh, min_area=2000, max_n_objs=5):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_list = sorted([
        (cv2.boundingRect(c),cv2.contourArea(c)) 
        for c in cnts ], 
        key= lambda x : -x[1])
    cnts = [ ca[0] for ca in cnt_list if ca[1] > min_area ]
    cnts = cnts[:min(max_n_objs, len(cnts))]
    return cnts




class YOLODetector:

    # These map the output type -> 
    VOC_LABELS = {
        1:'aeroplane',
        2:'bicycle',
        3:'bird',
        4:'boat',
        5:'bottle',
        6:'bus',
        7:'car',
        8:'cat',
        9:'chair',
        10:'cow',
        11:'diningtable',
        12:'dog',
        13:'horse',
        14:'motorbike',
        15:'person',
        16:'pottedplant',
        17:'sheep',
        18:'sofa',
        19:'train',
        20:'tvmonitor'
    }

    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.model = nets.TinyYOLOv2VOC(self.inputs) 
        self.sess = tf.Session()
        self.sess.run(self.model.pretrained())

    def apply(self, frame):
        frame = cv2.resize(frame, (416,416))
        frame_in = np.array(frame).reshape(-1,416,416,3)
        preds = self.sess.run(self.model, {self.inputs: self.model.preprocess(frame_in)})
        boxes = self.model.get_boxes(preds, frame_in.shape[1:3])
        cnts = []
        for i, box_type in enumerate(boxes):
            for r in box_type:
                print(f'{YOLODetector.VOC_LABELS[i+1]} at ({(r[0]+r[2])/2},{(r[1]+r[3])/2})')
                cnts.append((int(r[0]),int(r[1]),int(r[2]),int(r[3])))
        return cnts  
        

# TODO can we bootstrap / parameter search for a lightweight motion tracker
# via "transfer learning" of a powerful object detector? or at least generate
# data for validation (like iou rects
# TODO generate test frames with moving rect. we can *place* the rect and then
# estimate the detection (iou metrics n such).
# TODO find existing traffic / object dataset
# TODO test blender render to generate bboxes
