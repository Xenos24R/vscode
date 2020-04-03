import cv2.cv2 as cv
import numpy as np

def contours_demo(image):
    dst = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#阈值二值化处理
    cv.imshow("binary image",binary)

    contours,heriachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv.drawContours(image,contours,i,(0,0,255),2)#最后一个参数-1为填充
        print(i)
    cv.imshow("result",image)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
contours_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()