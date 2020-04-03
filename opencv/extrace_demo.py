import cv2.cv2 as cv
import numpy as np

def extrace_object_demo():#追踪目标颜色，二值化图像
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()
        frame = cv.flip(frame,1)
        if ret==False:
            break
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)#转化为HSV格式
        lower_hsv = np.array([37,43,46])#目标颜色像素范围
        higher_hsv = np.array([77,255,255])
        mask = cv.inRange(hsv,lower_hsv,higher_hsv)#过滤出目标颜色
        dst = cv.bitwise_and(frame,frame,mask=mask)#mask属性为掩膜
        cv.imshow("video",frame)
        cv.imshow("mask",dst)
        c = cv.waitKey(40)
        if c == 27:
            break

def split_object_demo():#cv.split():三通道分离;cv.merge():通道合并
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()
        if ret==False:
            break
        b,g,r = cv.split(frame)
        cv.imshow("image",frame)
        cv.imshow("blue",b)
        cv.imshow("green",g)
        cv.imshow("red",r)
        cv.waitKey(0)
        cv.destroyAllWindows()

extrace_object_demo()