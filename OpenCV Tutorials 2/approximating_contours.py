#Approximating Contours
import cv2
import numpy as np

image = cv2.imread('house.jpg')
original_image = image.copy()
cv2.imshow('Original Image', original_image)
cv2.waitKey()

#Gray scale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#Find contours
im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Iterating through each contour and computing the bounding rectangle
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow('Bounding Rectangle', original_image)

cv2.waitKey()

#Iterate through each contour and compute the approx contour
for c in contours:
    #Calculating accuracy as a percent of the contour perimeter
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow('Approx Poly DP', image)

cv2.waitKey()
cv2.destroyAllWindows()

#Convex Hull
image = cv2.imread('hand.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Image', image)
cv2.waitKey()

#Thresholding the image
ret, thresh = cv2.threshold(gray, 176, 255, 0)

#Finding the contours
im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Sorting contours by area and then remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

#Iterate through contours and draw the convex Hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow('Convex Hull', image)

cv2.waitKey()
cv2.destroyAllWindows()
