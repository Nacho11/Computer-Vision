import cv2
import numpy as np
import os
import sys

def chi2_distance(search_hist, hist):
    eps = 1e-10
    dist = 0.5 * np.sum([((a-b)**2) / (a+b+eps) for (a,b) in zip(search_hist, hist)])
    return dist

args = sys.argv
search_image = args[1] #Path to search image
search_image = cv2.imread(search_image)
cv2.imshow('Query', search_image)
cv2.waitKey()
cv2.destroyAllWindows()

image_name_image_feature_map = {}
for filename in os.listdir('images'):
    image = cv2.imread("images/"+filename)
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    print(hist)
    hist = cv2.normalize(hist, None)
    hist = hist.flatten()
    image_name_image_feature_map[filename] = hist

print(len(image_name_image_feature_map.keys()))

search_hist = cv2.calcHist([search_image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
search_hist = cv2.normalize(search_hist, None)
search_hist = search_hist.flatten()


distance_map = {}
inverse_dist_map = {}
for each_key in image_name_image_feature_map.keys():
    #compute the chi2_distance
    dist = chi2_distance(search_hist, image_name_image_feature_map[each_key])
    distance_map[each_key] = dist
    inverse_dist_map[dist] = each_key

print(distance_map)
sorted_images = [value for (key, value) in sorted(inverse_dist_map.items())]
print(sorted_images)
for each_image_name in sorted_images:
    image = cv2.imread('images/'+each_image_name)
    cv2.imshow(each_image_name , image)
    cv2.waitKey()
    cv2.destroyAllWindows()
