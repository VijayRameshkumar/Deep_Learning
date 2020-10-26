import cv2
import numpy as np

protoFile = "models/face/pose_deploy.prototxt"
weightsFile = "models/face/pose_iter_116000.caffemodel"
nPoints = 22

# Load Model and Image
frame = cv2.imread("python_api/images/priyanka_frontal.jpg")

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


for i in range(70):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    print(probMap.shape, "\n\n")
    probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        # cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(point[0]), int(point[1])))
    else:
        points.append(None)

cv2.imshow('Output-Keypoints', frameCopy)
if cv2.waitKey(0) == 27:
    cv2.destroyWindow("Output-Keypoints")
else:
    pass

cv2.imwrite("python_api/output/" + "priyanka_facial_keypoints.jpg", frameCopy)