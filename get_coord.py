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

img_original = cv2.imread('./img/book.png')

while True:
    cv2.imshow("original", img_original)
    height, weight = img_original.shape[:2]
    # press SPACEBAR to break
    if cv2.waitKey(1)&0xFF == 32:
        break

print("point_list:",point_list)

# coordinate order - upper left > upper right > lower left > lower right
pts_src = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
pts_dst = np.float32([[0,0], [weight,0], [0,height], [weight,height]])

(x, y) = point_list[-1]
_original_coord = np.array([[x, y]], dtype='float32')
original_coord = np.array([_original_coord])


M = cv2.getPerspectiveTransform(pts_src, pts_dst)
H, status = cv2.findHomography(pts_src, pts_dst)

transformed_coord = cv2.perspectiveTransform(original_coord, H)
img_result = cv2.warpPerspective(img_original, M, (weight, height))

print(pts_src)
print("original_coord:", original_coord)
print(pts_dst)
print("transformed_coord:", transformed_coord)

cv2.imshow("result", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()