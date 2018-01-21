import cv2
import numpy as np

image = cv2.imread('soduku.jpg')
image2 = cv2.imread('soduku.jpg')
#gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Canny Edges
edges = cv2.Canny(gray, 100, 170, apertureSize = 3)

#Run HoughLines using a rho accuracy of 1 pixel
#Theta accuracy of np.pi / 180 which is 1 degree
#Line threshold is set to 240
lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)
print(lines.shape)
#Iterate through each line and convert it to the format required by cv.lines
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2,y2), (255,0,0), 2)

cv2.imshow('Hough Lines', image)
cv2.waitKey()
cv2.destroyAllWindows()

# Probabilistic Hough Lines
hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 5, 10)
print(hough_lines.shape)
for each_line in hough_lines:
    print(each_line)
    for x1, y1, x2, y2 in each_line:
        cv2.line(image2, (x1,y1), (x2,y2), (0, 255, 0), 3)

cv2.imshow('Probabilistic Hough Lines', image2)
cv2.waitKey()
cv2.destroyAllWindows()
