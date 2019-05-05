# https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
import cv2  # import the OpenCV library
import numpy as np  # import the numpy library

# provide points from image 1
pts_src = np.array([[154, 174], [702, 349], [702, 572], [1, 572], [1, 191]])
# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
pts_dst = np.array([[212, 80], [489, 80], [505, 180], [367, 235], [144, 153]])

# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)
print(h)


# provide a point you wish to map from image 1 to image 2
a = np.array([[154, 174]], dtype='float32')
a = np.array([a])
print(a)

# finally, get the mapping
pointsOut = cv2.perspectiveTransform(a, h)

print(pointsOut)