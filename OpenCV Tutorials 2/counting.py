import cv2
import numpy as np

#Counting the number of blobs in an image
image = cv2.imread('blobs.jpg', 0)
cv2.imshow('Original Image', image)
cv2.waitKey()

#Initialize the detector using default parameters
detector = cv2.SimpleBlobDetector_create()

#Detect blobs
keypoints = detector.detect(image)

#Draw blobs
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)

cv2.imshow("Blobs using default parameters", blobs)
cv2.waitKey()

#Counting Circles and Ellipses

#Initialize parameter settings
params = cv2.SimpleBlobDetector_Params()

#Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

#Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9

#Set Convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create detector with params
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

blank  = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of circular blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey()
cv2.destroyAllWindows()
