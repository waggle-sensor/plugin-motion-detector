import time
from itertools import combinations
import cv2

class TrackedObject:
    def __init__(self, rect):
        self.rect = rect
        self.last_seen = time.time()

    def __str__(self):
        return str(self.rect)


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
        for obj in objs:
            new_rect = None 
            for rect in detected_rects:
                if _rect_center_overlap(rect, obj.rect):
                    new_rect = rect if new_rect is None else _union_rect(rect, new_rect)
                    seen = True
                    break
            if new_rect != None:
                obj.rect = tuple(
                            int(self.weight*new_rect[i] + (1.0-self.weight)*obj.rect[i])
                            for i in range(4))
                obj.last_seen = time.time()
                new_objs.append(obj)
            elif (time.time() - obj.last_seen) < self.object_ttl:
                new_objs.append(obj)
        
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
        
        # (attempt to) track objects into the next frame:
        self.tracker.update_objs(self.tracked_objs, frame, detected_rects)

    def show_tracked_objects(self, frame):
        for obj in self.tracked_objs:
            x,y,w,h = obj.rect
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
