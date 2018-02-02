import cv2
import numpy as np

image = cv2.imread('WaldoBeach.jpg')
cv2.imshow('Waldo', image)
cv2.waitKey()
cv2.destroyAllWindows()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = cv2.imread('waldo.jpg', 0)
cv2.imshow('Template', template)
cv2.waitKey()
cv2.destroyAllWindows()
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#Create the bounding box
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
