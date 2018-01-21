import cv2
import numpy as np

def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

#Returns the X coordinate for the contour centroid
def x_cord_contour(contours):
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))

#Places red circle on the centers of contours
def label_contour_center(image, c, i):
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(image, (cx,cy), 10, (0,0,255), -1)
    return image


image = cv2.imread('bunchofshapes.jpg')
cv2.imshow('Original', image)
cv2.waitKey()

#Creating a block image with same dimensions as our loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))

#Copy of original image
original_image = image.copy()

#Converting to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Find Canny Edges
canny = cv2.Canny(gray, 50, 200)
cv2.imshow('1 - Canny Edges', canny)
cv2.waitKey()

#Find contours and print how many were found
im2, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

#Draw all contours
cv2.drawContours(blank_image, contours, -1, (0,255,0), 3)
cv2.imshow('2 - All Contours over blank image', blank_image)
cv2.waitKey()

#Draw all contours over blank image
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('3 - All Contours', image)
cv2.waitKey()

cv2.destroyAllWindows()

#Sorting by area
#printing before sorting
print(get_contour_areas(contours))

#Sort contours large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#printing after sorting
print(get_contour_areas(sorted_contours))

#Iterate over contours and draw one at a time
for c in sorted_contours:
    cv2.drawContours(original_image, [c], -1, (255,0,0), 3)
    cv2.imshow('Contours by area', original_image)
    cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows()


#Sorting by position
for (i,c) in enumerate(contours):
    orig = label_contour_center(image, c, i)

cv2.imshow("4 - Contour Centers", image)
cv2.waitKey()

#Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)

#Labeling Contours left to right
for (i,c) in enumerate(contours_left_to_right):
    cv2.drawContours(original_image, [c], -1, (0,0,255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(original_image, str(i+1), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Left To Right Contour', original_image)
    cv2.waitKey()
    (x, y, w, h) = cv2.boundingRect(c)

    #Cropping the Contour
    cropped_contour = original_image[y:y+h, x:x+w]
    image_name = "output_shape_number_" + str(i+1) + ".jpg"
    print(image_name)
    cv2.imwrite(image_name, cropped_contour)

cv2.destroyAllWindows()        
