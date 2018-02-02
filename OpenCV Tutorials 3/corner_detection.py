#Harris Corner Detection
import cv2
import numpy as np

image = cv2.imread('chess.jpg')
gray_one = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#The cornerHarris function requires the array datatype to be float32
gray = np.float32(gray_one)

harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

#We use dilation of the corner points to enlarge them
kernel = np.ones((7,7), np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)

#Threshold for an optimal value, it may vary depending on the image
image[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]
cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Improved Corner detection - good features to track
img = cv2.imread('chess.jpg')
corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 15)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (0,255,0), 2)

cv2.imshow("Corners Found", img)
cv2.waitKey()
cv2.destroyAllWindows()
