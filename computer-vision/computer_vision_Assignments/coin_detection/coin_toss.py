import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

img = cv2.imread('water_coins.jpg', 1)
input = img.copy()

gray = cv2.imread('water_coins.jpg', 0)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 255]

dup = img.copy()

def crop_center(img, new_size):
    y, x, c = img.shape
    (cropx, cropy) = new_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]

height, width = markers.shape

new_image = []
for i in markers.flatten():
    if i == -1:
        new_image.append(255)
    else:
        new_image.append(0)

new_image = np.array(new_image).reshape(height, width)

cnts = cv2.findContours(new_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for contour in cnts:
    try:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cv2.contourArea(contour) > 100:
            # cv2.drawContours(dup, [contour], -1, (0, 255, 0), 2)
            cv2.circle(dup, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(dup, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    except:
        pass
cv2.imshow("input", input)
cv2.imshow("coin_detection", img)
cv2.imshow("output", dup)

cv2.imwrite("coin_detection.jpg", img)
cv2.imwrite("centroid.jpg", dup)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()