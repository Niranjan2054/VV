import numpy as np
class CommandIndex:
    def __init__(self,frame,roi,class_index):
        self.frame = frame
        self.roi = roi
        self.class_index = class_index
        self.height,self.width = self.frame.shape[:2]
    
    def returnIndex(self):
        return self.class_index * 3 + self.regionValue()
    
    def regionValue(self):
        left = [0,0,int(self.width/3),self.height]
        center = [int(self.width/3),0,int(self.width/3*2),self.height]
        right = [int(self.width/3*2),0,int(self.width),self.height]
        l = []
        l.append(self.intersection(center,self.roi))
        l.append(self.intersection(left,self.roi))
        l.append(self.intersection(right,self.roi))
        return l.index(max(l)) + 1

    def intersection(self,box1,box2):
        int_x1 = max(box1[0],box2[0])
        int_y1 = max(box1[1],box2[1])
        int_x2 = min(box1[2],box2[2])
        int_y2 = min(box1[3],box2[3])
        a = (int_x1-int_x2) * (int_y1*int_y2)
        if a>0:
            return a
        else: 
            return -a
