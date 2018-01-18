import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('sample.jpg')
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

#Ravel() flattens the image array
plt.hist(image.ravel(), 256, [0, 256])
plt.show()

color = ('b', 'g', 'r')

#We now separate the colors and plot each in the histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0, 256])

plt.show()     
