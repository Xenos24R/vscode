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