from imutils.video import FPS
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


filename = 'sample2'
cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

vs = cv2.VideoCapture('./videos/{}.mp4'.format(filename))

width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = FPS().start()
writer = None

# WRITE THE FIRST FRAME
success, image = vs.read()
if success:
    cv2.imwrite("./videos/{}.jpg".format(filename), image)

img_original = cv2.imread('./videos/{}.jpg'.format(filename))

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

(x, y) = point_list[-1]
_original_coord = np.array([[x, y]], dtype='float32')
original_coord = np.array([_original_coord])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
H, status = cv2.findHomography(pts_src, pts_dst)

cv2.destroyAllWindows()


# VIDEO
while True:
    (grabbed, frame) = vs.read()

    if frame is None:
        break

    # frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('./output/{}_perspective.avi'.format(filename), fourcc,
                                 30, (frame.shape[1], frame.shape[0]), True)

    # DO SOMETHING
    transformed_coord = cv2.perspectiveTransform(original_coord, H)
    transformed_frame = cv2.warpPerspective(frame, M, (width, height))


    if writer is not None:
        writer.write(transformed_frame)

    cv2.imshow("original", frame)
    cv2.imshow("transformed", transformed_frame)
    # press "q" to break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()
cv2.destroyAllWindows()
vs.release()
