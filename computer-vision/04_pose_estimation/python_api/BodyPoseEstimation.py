import cv2
import numpy as np
import matplotlib.pyplot as plt

protoFile = "models/pose/mpi/pose_deploy_linevec.prototxt"
weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

image = cv2.imread('python_api/images/yoga_pose2.jpg', -1)
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (image.shape[1], image.shape[0]), (0, 0, 0), swapRB=False, crop=False)
net.setInput(blob)
out = net.forward()

threshold = 0.1

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
             "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
             "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
             "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
             ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
             ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
             ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

points = []


for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponding body's part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv2.minMaxLoc(heatMap)
    x = (image.shape[1] * point[0]) / out.shape[3]
    y = (image.shape[0] * point[1]) / out.shape[2]

    # Add a point if it's confidence is higher than threshold.
    points.append((int(x), int(y)) if conf > threshold else None)

img = np.ones(shape=(image.shape[0], image.shape[1], 3))
cv2.imshow("test", img)
cv2.waitKey(0)

for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]
    if points[idFrom] and points[idTo]:
        cv2.line(img, points[idFrom], points[idTo], (255, 255, 255), 3)
        cv2.ellipse(img, points[idFrom], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)
        cv2.ellipse(img, points[idTo], (4, 4), 0, 0, 360, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) , 1, cv2.LINE_AA)
        cv2.putText(img, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow("frame", image)

if cv2.waitKey(0) == 27:
    cv2.destroyWindow("frame")
cv2.imshow("frame", img)
if cv2.waitKey(0) == 27:
    cv2.destroyWindow("frame")

cv2.imwrite("python_api/output/yoga_pose21.jpg", img)