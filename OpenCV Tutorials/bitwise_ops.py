import cv2
import numpy as np

#Making a square
square = np.zeros((300,300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow("Square", square)
cv2.waitKey()

#Making a ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("Ellipse", ellipse)
cv2.waitKey()

cv2.destroyAllWindows()

# Shows only where they intersect
bitwiseAnd = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey()

# Shows whether square or ellipse
bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey()

# Shows where either exists by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey()

# Shows everything that is not a suqare
bitwiseNotSq = cv2.bitwise_not(square)
cv2.imshow("NOT - square", bitwiseNotSq)
cv2.waitKey()

cv2.destroyAllWindows()
