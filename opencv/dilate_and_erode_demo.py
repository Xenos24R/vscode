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