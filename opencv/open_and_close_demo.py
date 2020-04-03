import cv2.cv2 as cv
import numpy as np

"""
去除小的干扰块-开操作
填充闭合区域-闭操作
水平或垂直线提取
"""
def open_demo(image):#开操作=腐蚀+膨胀，输入图像+结构元素
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))#当希望提取水平线时，使用长条结构元素（15，1），竖直的线则用（1，15）
    binary = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv.imshow("dst",binary)

def close_demo(image):#闭操作=膨胀+腐蚀，输入图像+结构元素。
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    binary = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)
    cv.imshow("dst",binary)

src = cv.imread("G:/openCV/opencv/sources/samples/data/tmpl.png")
open_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()