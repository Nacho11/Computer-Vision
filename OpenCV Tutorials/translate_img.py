import cv2
import numpy as np

image = cv2.imread('sample.jpg')
height, width = image.shape[:2]

quater_height, quater_width = height/4, width/4

T = np.float32([[1,0,quater_height], [0,1,quater_width]])
print(T)

#warpAffine transforms the image using the matrix T
img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translated Image', img_translation)
cv2.waitKey()
cv2.destroyAllWindows()

#Rotating the image around its center
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

#Another way
transpose_image = cv2.transpose(image)
cv2.imshow('Transpose Image', transpose_image)
cv2.waitKey()
cv2.destroyAllWindows()

#Resizing Image - making it 3/4th
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
cv2.imshow('Scaling - Linear Interpolation', image_scaled)
cv2.waitKey()
cv2.destroyAllWindows()

#doubling the size of the image
image_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', image_scaled)
cv2.waitKey()
cv2.destroyAllWindows()

#Skewing the re-sizing by setting exact dimensions
image_scaled = cv2.resize(image, (900, 400), interpolation=cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', image_scaled)
cv2.waitKey()
cv2.destroyAllWindows()

#Image Pyramids
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

cv2.imshow('Original', image)
cv2.imshow('Smaller Image', smaller)
cv2.imshow('Larger Image', larger)
cv2.waitKey()
cv2.destroyAllWindows()
