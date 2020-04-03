import cv2.cv2 as cv
import numpy as np

def video_demo():
    capture = cv.VideoCapture(0)
    while True:
        frame = capture.read()
        frame = cv.flip(frame,1)#镜像翻转
        cv.imshow("video",frame)
        c = cv.waitKey(50)
        if c == 27:
            break
video_demo()
