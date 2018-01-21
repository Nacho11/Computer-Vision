import cv2
import numpy as np

image = cv2.imread('Sunflowers.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(gray)

#Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blobs", blobs)
cv2.waitKey()
cv2.destroyAllWindows()
