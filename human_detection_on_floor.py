# USAGE
# python detection.py --input videos/sample1.mp4 --output output/sample1.avi --yolo yolo-coco

import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-y", "--yolo", default="yolo-coco", help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


point_list = []
count = 0
def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append((x, y))
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)


filename = args["input"][7:-4]
cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)
vs = cv2.VideoCapture(args["input"])
writer1, writer2 = None, None
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("W, H:", W, H)

# WRITE THE FIRST FRAME
success, image = vs.read()
if success:
    cv2.imwrite("./videos/{}.jpg".format(filename), image)

img_original = cv2.imread('./videos/{}.jpg'.format(filename))

while True:
    cv2.imshow("original", img_original)
    H, W = img_original.shape[:2]
    # press SPACEBAR to break
    if cv2.waitKey(1)&0xFF == 32:
        break

print("point_list:",point_list)
# coordinate order - upper left > upper right > lower left > lower right
pts_src = np.float32([list(point_list[0]), list(point_list[1]), list(point_list[2]), list(point_list[3])])
pts_dst = np.float32([[0,0], [W,0], [0,H], [W,H]])

pts = [list(point_list[0]), list(point_list[1]), list(point_list[3]), list(point_list[2])]
pts = np.array(pts)
pts = pts.reshape((-1, 1, 2))
space = np.int32(pts)


(x, y) = point_list[-1]
_original_coord = np.array([[x, y]], dtype='float32')
original_coord = np.array([_original_coord])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
HM, status = cv2.findHomography(pts_src, pts_dst)

cv2.destroyAllWindows()



try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x1 = int(centerX - (width / 2))
                y1 = int(centerY - (height / 2))
                x2 = int(centerX + (width / 2))
                y2 = int(centerY + (height / 2))

                if LABELS[classID] == 'person':
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    transformed_frame = cv2.warpPerspective(frame, M, (W, H))

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x1, y1) = (boxes[i][0], boxes[i][1])
            (x2, y2) = (boxes[i][2], boxes[i][3])

            original_1 = np.array([[[x1, y1]]], dtype='float32')
            original_2 = np.array([[[x2, y2]]], dtype='float32')

            transformed_1 = cv2.perspectiveTransform(original_1, HM)
            transformed_2 = cv2.perspectiveTransform(original_2, HM)

            t_x1, t_y1 = int(transformed_1[0][0][0]), int(transformed_1[0][0][1])
            t_x2, t_y2 = int(transformed_2[0][0][0]), int(transformed_2[0][0][1])

            x_center = int((x1 + x2) / 2)
            y_bottom = y2
            t_x_center = int((t_x1+t_x2)/2)
            t_y_bottom = t_y2

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.polylines(frame, [space], True, (0,255,0), 2)
            cv2.circle(frame, (x_center, y_bottom), 5, (0, 0, 255), -1)
            cv2.circle(transformed_frame, (t_x_center, t_y_bottom), 5, (0, 0, 255), -1)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    if writer1 is None and writer2 is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer1 = cv2.VideoWriter('output/{}_detect.avi'.format(filename), fourcc,
                                  30, (frame.shape[1], frame.shape[0]), True)
        writer2 = cv2.VideoWriter('output/{}_transform.avi'.format(filename), fourcc,
                                  30, (transformed_frame.shape[1], transformed_frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    writer1.write(frame)
    writer2.write(transformed_frame)

print("[INFO] cleaning up...")
writer1.release()
writer2.release()
vs.release()
print("[INFO] Finished!")
