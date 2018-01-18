import cv2
import numpy as np

def sketch(image):
    #Convert image to grey scale
    #img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Clean up image using Guassian Blur
    img_grey_blur = cv2.GaussianBlur(image, (5,5), 0)

    #Extract Edges
    canny_edges = cv2.Canny(img_grey_blur, 10, 70)

    #Invert Binarize the image
    ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
    return mask

#Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    #cv2.imshow('Sketch', frame)
    cv2.imshow('Our Live Sketcher', sketch(frame))
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()
