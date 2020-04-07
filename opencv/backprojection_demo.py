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
    mask = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],3)#直方图反向投影
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


"""
计算直方图
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]):
    image:输入图像，输入时要写在中括号里
    channels:传入的通道，如果是灰度图像则值为零，如果是彩色图像则对应的是BGR或HSV
    mask:掩膜
    histSize:灰度级个数，输入时需要中括号，如[256]表示有256个灰度级
    ranges[]:取值范围，通常是[0,256]，每一个histSize对应ranges中的两个值(min,max)

归一化
cv2.normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]):
    src:输入的图像
    dst[]:与src大小相同的输出结果
    alpha[]:
    beta[]:
    norm_type:NORM_INF,NORM_L1,NORM_L2,,NORM_MINMAX
    mask:掩膜

直方图反向投影
cv2.calcBackProject(images, channels, hist, ranges, scale[, dst])
    image:输入的图片
    channels:用于计算反向投影的通道列表，通道数必须与直方图维度相匹配
    hist:输入的直方图
    ranges:直方图每个通道的bin取值范围
    scale[]:可选输出反向投影的比例因子，取值越大反向投影的色度范围就越大
"""