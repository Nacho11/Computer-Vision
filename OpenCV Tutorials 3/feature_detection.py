#This is OpenCV version 2 and not 3 - in 3 SIFT and SURF are not supported

import cv2
import numpy as np


#SIFT
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#create a sift feature detector
sift = cv2.SIFT()

#Detect key points
keypoints = sift.detect(gray, None)
#print("Number of keypoints detected: " + len(keypoints))

#Draw rich key points on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - SIFT', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#SURF
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Create SURF feature detector object
surf = cv2.SURF()

# Only features, whose hessian is larger than hessianThreshold are retained by the detector
surf.hessianThreshold = 7500
keypoints, descriptors = surf.detectAndCompute(gray, None)

image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SURF', image)
cv2.waitKey()
cv2.destroyAllWindows()


#FAST
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST Detector object
fast = cv2.FastFeatureDetector()
#print(fast)
# Obtain Key points, by default non max suppression is On
# to turn off set fast.setBool('nonmaxSuppression', False)
#fast.setBool('nonmaxSuppression', 0)
keypoints = fast.detect(gray, None)
print("Number of keypoints detected: ", len(keypoints))

# Draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - FAST', image)
cv2.waitKey()
cv2.destroyAllWindows()

#BRIEF
#Create BRIEF extractor object
brief = cv2.DescriptorExtractor_create('BRIEF')

#Determine key points
keypoints = fast.detect(gray, None)

#Obtain descriptors and new final keypoints using BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)
print("Number of keypoints detected: ", len(keypoints))

image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - BRIEF', image)
cv2.waitKey()
cv2.destroyAllWindows()

#ORB - Oriented FAST and Rotated BRIEF (ORB)
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB(10)

#Determine key points
keypoints = orb.detect(gray, None)

#Obtain the descriptors
keypoints, descriptors = orb.compute(gray, keypoints)
print("Number of keypoints detected: ", len(keypoints))

#Draw rich keypoints on input image
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - ORB', image)
cv2.waitKey()
cv2.destroyAllWindows()






'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('input.jpg',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
#print "Threshold: ", fast.getInt('threshold')
#print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
#print "neighborhood: ", fast.getInt('type')
#print "Total Keypoints with nonmaxSuppression: ", len(kp)

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

#print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)
'''
'''
image = cv2.imread('input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Create FAST detector object
fast = cv2.FastFeatureDetector()

#Create a brief extractor object
brief = cv2.FastFeatureDetector()

#Obtain descriptors and new finalkeypoints using BRIEF
keypoints, descriptors = brief.compute(gray)

image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - BRIEF', image)
cv2.waitKey()
cv2.destroyAllWindows()
'''
