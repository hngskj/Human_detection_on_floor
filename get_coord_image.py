import numpy as np
import cv2

point_list = []
count = 0

def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append((x, y))
        if count < 4:
            cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)
        else:
            cv2.circle(img_original, (x, y), 3, (0, 255, 0), -1)
        count += 1

cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

img_original = cv2.imread('./images/book.png')
height, width = img_original.shape[:2]
print("W, H:", width, height)

while True:
    cv2.imshow("original", img_original)
    height, width = img_original.shape[:2]
    # press SPACEBAR to break
    if cv2.waitKey(1)&0xFF == 32:
        break

print("point_list:",point_list)

# coordinate order - upper left > upper right > lower left > lower right
pts_src = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
pts_dst = np.float32([[0,0], [width,0], [0,height], [width,height]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
H, status = cv2.findHomography(pts_src, pts_dst)


(x, y) = point_list[-1]
print("point:", x, y)
original_coord = np.array([[[x, y]]], dtype='float32')
print("original_coord:", original_coord)

transformed_coord = cv2.perspectiveTransform(original_coord, H)
print("transformed_coord:", transformed_coord)


img_result = cv2.warpPerspective(img_original, M, (width, height))
# print(pts_src)
# print(pts_dst)
cv2.imshow("result", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()