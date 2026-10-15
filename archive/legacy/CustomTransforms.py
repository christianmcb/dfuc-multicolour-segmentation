import cv2
import numpy as np
from albumentations.core.transforms_interface import *
        
class CustomSharpen(ImageOnlyTransform):
	def apply(self, img, **params):
		return cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
	
class CustomUpSample(DualTransform):
	def apply(self, img, **params):
		return cv2.pyrUp(img)
		
class CustomHSV(ImageOnlyTransform):
	def apply(self, img, **params):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV colourspace
		#img = np.divide(img, np.array([0.70196078431, 1, 1])) # convert range back to [255,255,255]
		return img
