import cv2
import numpy as np

image = cv2.imread('shapes_donut.jpg')
cv2.imshow('Input Image', image)
cv2.waitKey()

#Make the image grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find the canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey()

#Find Contours
im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey()

print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', image)
cv2.waitKey()
cv2.destroyAllWindows()
