import cv2.cv2 as cv
import numpy as np

def equalHist_demo(image):#直方图均衡化
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("image",gray)
    cv.imshow("result",dst)

def clahe_demo(image):#局部置信的直方图均衡化
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.6,tileGridSize=(8,8))
    dst = clahe.apply(gray)
    cv.imshow("image",gray)
    cv.imshow("result",dst)
    
src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
clahe_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
