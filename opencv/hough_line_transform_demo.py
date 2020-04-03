import cv2.cv2 as cv
import numpy as np

def haugh_line_transform_demo(image):#霍夫直线变换，利用极坐标，需要添加画图语句
    #dst = cv.pyrMeanShiftFiltering(image,10,50)霍夫变换对噪声敏感，可先降噪，再霍夫直线变换
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray,50,150,apertureSize=3)
    lines = cv.HoughLines(canny,1,np.pi/180,250)#返回值是极坐标下的ρ和θ
    print(lines)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv.imshow("result",image)

def line_detect_possible_demo(image):#霍夫直线变换，自动画线
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray,50,150,apertureSize=3)
    lines = cv.HoughLinesP(edge,1,np.pi/180,200,minLineLength=50,maxLineGap=10)#返回值是线段始末点坐标
    print(lines)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imshow("result",image)


src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
haugh_line_transform_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
