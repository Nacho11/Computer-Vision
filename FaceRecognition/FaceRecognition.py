
import cv2
import face_recognition
import os
import imutils
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input directory of faces" )
ap.add_argument("-t", "--test_image", required=True, help="test image to detect faces")
args = vars(ap.parse_args())

dataset = args["dataset"]
test_image = args["test_image"]

knownEncodings = []
knownNames = []

for foldername in os.listdir(dataset):
    if foldername == '.DS_Store':
        continue
    for each_image in os.listdir(dataset + '/' + foldername):
        if each_image == '.DS_Store':
            continue
        knownNames.append(foldername[:-5])
        image = cv2.imread('dataset/' + foldername + '/' + each_image)
        image = cv2.resize(image, (256, 256))
        image = np.array(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        knownEncodings.append(encodings)

image = cv2.imread(test_image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)

encoding_index_matches_name_map = {}
for j in range(0, len(encodings)):
    names_count_map = {}
    true_index = []
    for i in range(0, len(knownEncodings)):
        match = face_recognition.compare_faces(knownEncodings[i], encodings[j])
        if len(match) > 0 and match[0] == True:
            true_index.append(i)
            if knownNames[i] in names_count_map:
                names_count_map[knownNames[i]] += 1
            else:
                names_count_map[knownNames[i]] = 1
    max_count = 0
    name = ""
    for each_name in names_count_map:
        if max_count < names_count_map[each_name]:
            max_count = names_count_map[each_name]
            name = each_name
    encoding_index_matches_name_map[j] = name

for i in range(0, len(encodings)):
    box = boxes[i]
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, encoding_index_matches_name_map[i], (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

plt.imsave("Results/" + test_image, image)
