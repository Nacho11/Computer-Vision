import cv2
import numpy as np
import sys

def ORB_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    #Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB(1000, 1.2)

    #Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)

    #Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    #Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Do matching
    matches = bf.match(des1, des2)

    #Sort the matches based on distance, least distance is better
    matches = sorted(matches, key=lambda val: val.distance)

    return len(matches)


def sift_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of SIFT matches between them

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template

    #Create SIFT detector object
    sift = cv2.SIFT()

    #Obtain keypoints and descriptors using SFIT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    #Define parameters for our Flann matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)

    #Create the flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #Obtain matches using k-nearest neighbor Method
    #the result matches is the similar number of matches found in both the images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store good matches using lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return len(good_matches)

type_of_detection = sys.argv[1]

cap = cv2.VideoCapture(0)

#Load our image template, this is our reference image
image_template = cv2.imread('diet_coke.jpg', 0)

while True:
    #Get webcam images
    ret, frame = cap.read()

    #Get height and width of webcam frame
    height, width = frame.shape[:2]

    #Define ROI Box Dimensions
    top_left_x = width / 3
    top_left_y = (height/2) + (height/4)
    bottom_right_x = (width/3) * 2
    bottom_right_y = (height/2) - (height/4)

    #Draw rectangular window for our region of interest
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)

    #Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

    #Flip frame orientation horizontally
    frame = cv2.flip(frame, 1)

    if type_of_detection == 0:
        #Get number of SIFT matches
        matches = sift_detector(cropped, image_template)
        threshold = 10
        text = 'SIFT'
    else:
        matches = ORB_detector(cropped, image_template)
        threshold = 400
        text = 'ORB'

    #Display status string showing the current number of matches
    cv2.putText(frame, str(matches), (450,450), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 1)

    #If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,255,0), 3)
        cv2.putText(frame, 'Object Found', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)


    cv2.imshow('Object Detector using ' + text, frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
