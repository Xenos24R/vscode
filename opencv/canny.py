import cv2.cv2 as cv
import numpy as np

def edge_demo(image):#边缘提取
    blurred = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad = cv.Sobel(gray,cv.CV_16SC1,0,1)
    edge = cv.Canny(xgrad,ygrad,50,150)
    dst = cv.bitwise_and(image,image,mask=edge)
    cv.imshow("edge",dst)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
edge_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()


"""
高斯模糊
cv2.GaussianBlur(src, ksize[], sigmaX, dst[], sigmaY, borderType)
    src:输入的图像
    ksize:模板大小
    sigmaX[]:高斯核函数在X方向上的标准偏差
    sigmaY[]:高斯核函数在Y方向上的标准偏差，如果为0，则与sigmaX相同
    borderType:用于推断图像外部像素的某种边界模式

Sobel算子
cv2.Sobel(src, ddepth, dx, dy[], dst[], ksize[], scale[], delta[], borderType)
    src:输入的图像
    ddepth:目标图像深度，当ddepth输入值为-1时，目标图像和原图像深度保持一致。图像深度是指存储每个像素所用的位数，也用于量度图像的色彩分辨率
    dx:x方向上求导的阶数
    dy:y方向上求导的阶数
    dst[]:与src大小相同的输出结果
    ksize[]:模板大小
    scale[]:缩放导数的比例常数，默认情况下没有伸缩系数
    delta[]:在储存目标图像前可选的添加到像素的值，默认值为0
    borderType:用于推断图像外部像素的某种边界模式

cv2.Canny(image, threshold1, threshold2[], edges[], apertureSize[], L2gradient)
    image:输入的图像
    threshold1:阈值1（较低）
    threshold2:阈值2（较高）
    edges[]:输出的边缘信息
    apertureSize[]:算子的大小
    L2gradient:是否使用L2范数（为0时使用L1范数）
"""