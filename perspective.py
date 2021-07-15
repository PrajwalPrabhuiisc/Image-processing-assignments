import cv2
import numpy as np

## To read the image
image = cv2.imread('cards.jpg')
## Using paint find the 4 corners of the image which needs perspective transform
ptD = [278, 25]
ptC = [507, 88]
ptA = [181, 352]
ptB = [418, 418]
##Specify the width and the height of transformed image (Users choice)
width, height = 500, 500
##Specify the points for the perspective transform as per method specified
pts1 = np.float32([ptD, ptC, ptA, ptB])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
##Apply the perspective transform Method
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (width, height))
##Show the result
cv2.imshow('Original', image)
cv2.imshow('Perspective', result)
cv2.waitKey()
