import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.jpg')
rows, cols = image.shape[:2]
cv2.imshow('Original', image)
cv2.waitKey()

#cordinates of the 4 corners of the image
points_A = np.float32([[320, 15], [700, 215], [85, 610], [530, 780]])

#coordinates of the 4 corners of the desired output
points_B = np.float32([[0,0], [420,0], [0, 594], [420, 594]])

#Perspective Transformation Matrix, M
M = cv2.getPerspectiveTransform(points_A, points_B)

pers_warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow('warpPerspective', pers_warped)
cv2.waitKey()
cv2.destroyAllWindows()

#Affine Transform
points_A = np.float32([[320, 15], [700, 215], [85, 610]])

points_B = np.float32([[0,0], [420,0], [0, 594]])

M = cv2.getAffineTransform(points_A, points_B)

aff_warped = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow('warpPerspective', aff_warped)
cv2.waitKey()
cv2.destroyAllWindows()
