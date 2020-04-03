import cv2.cv2 as cv
import numpy as np

def bilateral_filter_demo(image):#高斯双边模糊，除了考虑图像空间分布还考虑了图像的边缘
    dst = cv.bilateralFilter(image,0,100,10)
    cv.imshow("result",dst)

def mean_shift_demo(image):#，sigma代表着离散程度
    dst = cv.pyrMeanShiftFiltering(image,10,50)
    cv.imshow("result",dst)

src = cv.imread("C:/Users/32936/Desktop/2/3.jpg")
cv.imshow("image",src)
mean_shift_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()