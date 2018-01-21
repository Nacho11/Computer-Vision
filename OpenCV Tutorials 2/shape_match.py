import cv2
import numpy as np

template = cv2.imread('4star.jpg', 0)
cv2.imshow('Template', template)
cv2.waitKey()

#Load the target with the shapes to match
target = cv2.imread('shapestomatch.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# thresholding
ret1, thresh1 = cv2.threshold(template, 127, 255, 0)
ret2, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

#Find contours in template
im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#Removing the image outline
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#extract the template contour
template_contour = contours[1]

#Extract contours from second target image
im2, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    #Iterate through each contour in the target and compare contour shapes using cv2.matchShapes
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    #IF the match value is less than 0.15 we
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []

cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()
