import cv2
import numpy as np

class Feature:
    keypoint = cv2.KeyPoint(0, 0, 20)
    descriptor = np.array([]).astype('float32')
    orientation = 0
    
    
    def __init__(self, keypoint, descriptor, orientation):
        self.keypoint = keypoint
        self.descriptor = descriptor
        self.orientation = orientation
        
    def getFeature(self):
        return self.keypoint
    
    def getDescriptor(self):
        return self.descriptor
    
    def getOrientation(self):
        return self.orientation
    
    def getY(self):
        return self.keypoint.pt[1]
    
    def getX(self):
        return self.keypoint.pt[0]
    
    def equal(self, otherFeatureObject):
        return (np.array_equal(self.descriptor, otherFeatureObject.descriptor) and
                self.orientation == otherFeatureObject.orientation and
                self.getY() == otherFeatureObject.getY() and
                self.getX() == otherFeatureObject.getX()
                )