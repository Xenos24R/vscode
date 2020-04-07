import cv2.cv2 as cv
import numpy as np

#边缘保留滤波(EPF)常用的两个方法：高斯双边滤波和均值迁移滤波

def bilateral_filter_demo(image):#高斯双边模糊，除了考虑图像空间分布还考虑了图像的边缘
    dst = cv.bilateralFilter(image,0,1000,10)
    cv.imshow("result",dst)

def mean_shift_demo(image):#，sigma代表着离散程度
    dst = cv.pyrMeanShiftFiltering(image,10,50)
    cv.imshow("result",dst)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
cv.imshow("image",src)
bilateral_filter_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()


"""
高斯双边
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[], dst[], borderType)
    src:输入的图像
    d:滤波直径
    sigmaColor:颜色空间中的标准偏差
    sigmaSpace:协调空间中的标准偏差

均值迁移
cv2.pyrMeanShiftFiltering(src, sp, sr[], dst[], maxLevel[], termcrit)
    src:输入的图像
    sp:空间窗的半径（空间上的范围）
    sr:色彩窗的半径（色彩空间的范围）
    maxLevel:金字塔的最大层数
    termcrit:迭代终止的条件
"""