import cv2
import numpy as np

image = cv2.imread('sample.jpg', 0)
cv2.imshow("Original", image)

ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold', thresh1)

ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse', thresh2)

ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('Thresh Trunc', thresh3)

ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('Threshold zero', thresh4)

ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('Threshold zero inv', thresh5)
cv2.waitKey()
cv2.destroyAllWindows()

blur_img = cv2.GaussianBlur(image, (3, 3), 0)

#adaptive threshsold
adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
cv2.imshow("Adaptive Mean Thresholding", adaptive_threshold)
cv2.waitKey()

#Otsu's Thresholding
_, otsus_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Threshold", otsus_threshold)
cv2.waitKey()

cv2.destroyAllWindows()
