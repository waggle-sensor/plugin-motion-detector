import time
from itertools import combinations
import logging
import cv2

class TrackedObject:
    def __init__(self, rect, label='object'):
        self.label = label
        self.rect = rect
        self.first_seen = time.time()
        self.last_seen = self.first_seen

    def __str__(self):
        return f'{self.label} at: {str(self.rect)}'


def _rect_overlap(rA, rB):
    dx_x2 = abs(2*(rA[0] - rB[0]) + (rA[2] - rB[2]))
    dy_x2 = abs(2*(rA[1] - rB[1]) + (rA[3] - rB[3]))
    return (dx_x2 <= rA[2]+rB[2]) and (dy_x2 <= rA[3]+rB[3])    

def _rect_center_overlap(rA, rB): 
    dx_x2 = abs(2*(rA[0] - rB[0]) + (rA[2] - rB[2]))
    dy_x2 = abs(2*(rA[1] - rB[1]) + (rA[3] - rB[3]))
    return ((dx_x2 <= rA[2]) and (dy_x2 <= rA[3])) or \
           ((dx_x2 <= rB[2]) and (dy_x2 <= rB[3]))

def _union_rect(rA, rB):
    x = min(rA[0],rB[0])
    y = min(rA[1],rB[1])
    w = max(rA[0]+rA[2],rB[0]+rB[2])-x
    h =  max(rA[1]+rA[3],rB[1]+rB[3])-y
    return (x,y,w,h)


class EMATracker:
    def __init__(self, weight=0.5, object_ttl=1.0):
        self.weight = weight
        self.object_ttl = object_ttl

    def update_objs(self, objs, frame, detected_rects):
        new_objs = []

        # update old objects:
        for i, obj in enumerate(objs):
            new_rect = None 
            for rect in detected_rects:
                if _rect_center_overlap(rect, obj.rect):
                    new_rect = rect if new_rect is None else _union_rect(rect, new_rect)
                    break
            if new_rect != None:
                obj.rect = tuple(
                            int(self.weight*new_rect[i] + (1.0-self.weight)*obj.rect[i])
                            for i in range(4))
                
                # ensure object is not occluded by others:
                occluded = False
                for j, oth_obj in enumerate(objs[:i]):
                    if _rect_center_overlap(oth_obj.rect, obj.rect):
                        occluded = True
                        break
                if occluded:
                    logging.info(f'Merged objects #{j+1} and #{i+1}')
                else:
                    obj.last_seen = time.time()
                    new_objs.append(obj)
            
            # if the object is not in motion, wait for its ttl to expire:
            elif (time.time() - obj.last_seen) < self.object_ttl:
                new_objs.append(obj)
            else:
                logging.info(f'Lost object #{i+1}')
        
        # discover new objects:
        for rect in detected_rects:
            has_obj = False
            for obj in new_objs:
                if _rect_overlap(rect, obj.rect):
                    has_obj = True
                    break
            if not has_obj:
                new_objs.append(TrackedObject(rect))

        objs[:] = new_objs
        
class TrackedObjectDatabase:
    
    def __init__(self, detector, tracker, frame_buffer_size=2000):
        self.detector = detector
        self.tracker = tracker
        self.tracked_objs = []
    
    def update_tracked_objects(self, frame):
        detected_rects = self.detector.apply(frame)
        self.tracker.update_objs(self.tracked_objs, frame, detected_rects)

    def show_tracked_objects(self, frame, count=True):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, obj in enumerate(self.tracked_objs):
            x,y,w,h = obj.rect
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f'{i+1}', (x,y-20),font, 1, (0,0,255),2)
    
    def get_tracked_objects_info(self, with_meta=False):
        obj_dict = { 
            str(id(obj)) : { 'label': obj.label, 'rect': obj.rect, 'last_seen': obj.last_seen } 
            for obj in self.tracked_objs 
        }
        if with_meta:
            obj_meta = { 
                'detector': self.detector.__class__.__name__, 
                'tracker': self.tracker.__class__.__name__
            }
            return obj_dict, obj_meta
        
        return obj_dict
