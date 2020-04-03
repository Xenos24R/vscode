import cv2.cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plot_demo(image):#亮度直方图
    plt.hist(image.ravel(),256,[0,256])#ravel()降维
    plt.show("直方图")

def image_hist(image):#BGR直方图
    color = ('blue','green','red')
    for i,color in enumerate(color):
        hist = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()

src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
cv.imshow("img",src)
image_hist(src)