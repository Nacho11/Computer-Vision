import cv2
import numpy as np

image = cv2.imread('sample.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey()
kernel_3x3 = np.ones((3,3), np.float32) / 9

#Convolution operation
blurred_1 = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred_1)
cv2.waitKey()

kernel_7x7 = np.ones((7,7), np.float32) / 49

blurred_2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Bluring', blurred_2)
cv2.waitKey()

cv2.destroyAllWindows()

blur = cv2.blur(image, (3,3))
cv2.imshow('Averaging', blur)
cv2.waitKey()

Gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey()

median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey()

bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey()
cv2.destroyAllWindows()

#Sharpening
kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow("Image Sharpening", sharpened)
cv2.waitKey()
cv2.destroyAllWindows()
