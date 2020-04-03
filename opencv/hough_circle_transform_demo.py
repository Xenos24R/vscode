import cv2.cv2 as cv
import numpy as np

def haugh_circle_transform_demo(image):#霍夫圆检测
    dst = cv.pyrMeanShiftFiltering(image,10,100)#对噪声敏感，所以降噪
    cimage = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0,:]:#这里报错的原因不知道
        cv.circle(image,(i[0],i[1]),i[2],(0,0,0),2)
        cv.circle(image,(i[0],i[1]),2,(255,0,0),2)
    cv.imshow("circles",image)

src = cv.imread("G:/openCV/opencv/sources/samples/data/smarties.png")
haugh_circle_transform_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()