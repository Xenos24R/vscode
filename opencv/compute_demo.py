import cv2.cv2 as cv
import numpy as np
import operator as op
from functools import reduce

def add_demo(m1,m2):#加法
    dst = cv.add(m1,m2)
    cv.imshow("add result",dst)

def subtract_demo(m1,m2):#减法
    dst = cv.subtract(m1,m2)
    cv.imshow("sub result",dst)

def divide_demo(m1,m2):#除法
    dst = cv.divide(m1,m2)
    cv.imshow("divide result",dst)

def multiply_demo(m1,m2):#乘法
    dst = cv.multiply(m1,m2)
    cv.imshow("multiply result",dst)

def logic_demo(m1,m2):
    dst1 = cv.bitwise_and(m1,m2)
    dst2 = cv.bitwise_or(m1,m2)
    dst3 = cv.bitwise_not(m1)
    cv.imshow("and",dst1)
    cv.imshow("or",dst2)
    cv.imshow("not",dst3)

def others_demo(image):
    mean = cv.mean(image)#cv.mean()：均值
    meanStdDev = cv.meanStdDev(image)#cv.meanStdDev():方差，返回2darray，【1】是均值，【2】是方差
    #dst1 = cv.subtract(image,mean)
    print("mean:")
    print(mean)
    print("meanStdDev:")
    print(meanStdDev)

def contrast_brightness_demo(image,c,b):
    h,w,ch = image.shape
    blank = np.zeros([h,w,ch],image.dtype)
    dst = cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("demo",dst)

src1 = cv.imread("G:/openCV/opencv/sources/samples/data/LinuxLogo.jpg")
src2 = cv.imread("G:/openCV/opencv/sources/samples/data/WindowsLogo.jpg")

contrast_brightness_demo(src2,1.2,10)
cv.waitKey(0)
cv.destroyAllWindows()
