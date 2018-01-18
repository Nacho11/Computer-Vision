import cv2
import numpy as np

#Loading the image
input = cv2.imread('sample.jpg')
#Display the image
cv2.imshow('Sample Image', input)

#waitkey allows us to input information when a image window is open
#by placing the number we can specify the delay for how long to keep the window open
cv2.waitKey()
#This closes all open windows
cv2.destroyAllWindows()

print(input.shape)
#Use imwrite specifing the file name and the image to be saved
cv2.imwrite('output.jpg', input)

grey_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', grey_image)
cv2.waitKey(2000)
cv2.destroyAllWindows()

B, G, R = input[0, 0]
print(B, G, R)

print(grey_image[0,0])


#HSV Color Space
hsv_image = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_image)
cv2.imshow('Hue channel', hsv_image[:, :, 0])
cv2.imshow('Saturation channel', hsv_image[:,:, 1])
cv2.imshow('Value channel', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()

#Split function splites the image into each color index
B, G, R = cv2.split(input)
print(B.shape)
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey()
cv2.destroyAllWindows()

#Merging the images
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

#Amplify the blue color
merged = cv2.merge([B+100, G, R])
cv2.imshow("Blue Amplified", merged)

cv2.waitKey()
cv2.destroyAllWindows()

#Coloring the images
zeros = np.zeros(input.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

cv2.waitKey()
cv2.destroyAllWindows()
