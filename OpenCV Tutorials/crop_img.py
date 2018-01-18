import cv2
import numpy as np

image = cv2.imread('sample.jpg')
height, width = image.shape[:2]

#Let's get the starting pixel coordinates
start_row, start_col = int(height*0.40), int(width*0.30)

#Let's get the ending pixel coordinates
end_row, end_col = int(height*0.85), int(width*0.75)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("Original Image", image)
cv2.waitKey()
cv2.imshow("Cropped Image", cropped)
cv2.waitKey()
cv2.destroyAllWindows()

#Arithmetic Operations
#Create a matrix of ones, then multiply it by a scalar of 100
#This gives a matrix with same dimensions of our image will all values being 100
M = np.ones(image.shape, dtype='uint8') * 175

#We use this to add this matrix M, to our image
#Notice the increase in brightness
added = cv2.add(image, M)
cv2.imshow("Added", added)

#Likewise we can also subtract
#Notice the decrease in brightness
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtract", subtracted)

cv2.waitKey()
cv2.destroyAllWindows()
