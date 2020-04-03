import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def back_projection_demo():#反向投影
    roi = cv.imread("C:/Users/32936/Desktop/2/qiuyi.jpg")#搜索目标
    target = cv.imread("C:/Users/32936/Desktop/2/maidi.jpg")#待搜索图片
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    roihist = cv.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])#获取搜索目标直方图
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)#直方图归一化
    mask = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)#直方图反向投影
    dst = cv.bitwise_and(target,target,mask=mask)
    cv.imshow("result",dst)
    cv.imshow("target",target)
    cv.imshow("image",roi)
    cv.imshow("mask",mask)
def hist2d_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    #cv.imshow("2d",hist)
    plt.imshow(hist,interpolation='nearest')
    plt.title("2D")
    plt.show()

back_projection_demo()
cv.waitKey(0)