import cv2.cv2 as cv
import numpy as np

def measure_object(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    print("threshold value:%s"%ret)
    cv.imshow("binary image",binary)
    dst = cv.cvtColor(binary,cv.COLOR_GRAY2BGR)
    contours,hireachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        area = cv.contourArea(contour)
        x,y,w,h = cv.boundingRect(contour)
        mm = cv.moments(contour)
        type(mm)
        if mm['m00']:
            cx = mm['m10']/mm['m00']
            cy = mm['m01']/mm['m00']
        else:
            continue
        cv.circle(dst,(np.int(cx),np.int(cy)),3,(0,0,255),-1)
        #cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)绘制矩形边界
        print("area",area)
        approxCurve = cv.approxPolyDP(contour,4,True)#多边形逼近
        if approxCurve.shape[0] > 6:#边数大于6条的
            cv.drawContours(dst,contours,i,(0,0,255),2)
        if approxCurve.shape[0] == 4:#矩形
            cv.drawContours(dst,contours,i,(0,255,0),2)
        if approxCurve.shape[0] == 3:#三角形
            cv.drawContours(dst,contours,i,(255,0,0),2)
    cv.imshow("image",dst)
    cv.imshow("souce",image)

src = cv.imread("G:/openCV/opencv/sources/samples/data/pic5.png")
measure_object(src)
cv.waitKey(0)
cv.destroyAllWindows()