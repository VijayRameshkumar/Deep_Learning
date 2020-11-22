import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

video = cv2.VideoCapture("cointoss.mp4")

while True:
    ret, img = video.read()
    if ret:
        input = img.copy()

        gray = cv2.imread('water_coins.jpg', 0)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)