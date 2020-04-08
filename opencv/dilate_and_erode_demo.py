import cv2.cv2 as cv
import numpy as np
#膨胀和腐蚀结果会像结构元素靠拢，支持单通道和多通道
def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#根据情况选择THRESH_BINARY或是TTHRESH_BINARY_INV
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.erode(binary,kernel)
    cv.imshow("erode",dst)
    cv.imshow("binary",binary)

def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#根据情况选择THRESH_BINARY或是TTHRESH_BINARY_INV
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.dilate(binary,kernel)
    cv.imshow("erode",dst)
    cv.imshow("binary",binary)

src = cv.imread("G:/openCV/opencv/sources/samples/data/tmpl.png")
erode_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()


"""
定义结构元素
cv2.getStructuringElement(shape, ksize[], anchor)
    shape:结构元素的形状（MORPH_RECT，MORPH_CROSS，MORPH_ELLIPSE）
    ksize:结构元素大小
    anchor:锚点位置
    ->retval:结构元素

腐蚀
cv2.erode(src, kernel[], dst[], anchor[], iterations[], borderType, borderValue)
    src:输入的图像
    kernel[]:处理所用的结构元素
    anchor[]:锚点位置
    borderType:用于推断图像外部像素的某种边界模式
    borderValue:边缘值
    ->dst:与src大小相同的输出结果

膨胀
cv2.dilate(src, kernel[], dst[], anchor[], iterations[], borderType, borderValue)
    src:输入的图像
    kernel[]:处理所用的结构元素
    anchor[]:锚点位置
    borderType:用于推断图像外部像素的某种边界模式
    borderValue:边缘值
    ->dst:与src大小相同的输出结果
"""