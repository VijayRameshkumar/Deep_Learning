import cv2
import numpy as np

protoFile = "models/hand/pose_deploy.prototxt"
weightsFile = "models/hand/pose_iter_102000.caffemodel"
nPoints = 22

# Load Model and Image
frame = cv2.imread("python_api\images\woman-hand-smartphone-desk.webp")
frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
print(frame.shape)

frameCopy = frame.copy()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Get prediction

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frame.shape[1], frame.shape[0]),
                          (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)

output = net.forward()

points = []

threshold = 0.1


for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    print(probMap.shape, "\n\n")
    probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else:
        points.append(None)

cv2.imshow('Output-Keypoints', frameCopy)
if cv2.waitKey(0) == 27:
    cv2.destroyWindow("Output-Keypoints")
else:
    pass

cv2.imwrite("python_api/output/" + "keypoints.jpg", frameCopy)

POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4],
              [0, 5], [5, 6], [6, 7], [7, 8], [6, 8],
              [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16],
              [0, 17], [17, 18], [18, 19], [19, 20]]
print(points)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Output-Skeleton', frame)
cv2.imwrite("python_api/output/skeleton.jpg", frame)

if cv2.waitKey(0) ==27:
    cv2.destroyWindow("Output-Skeleton")
else:
    pass