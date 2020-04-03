import cv2.cv2 as cv
import numpy as np

def large_image_demo(image):
    print(image.shape)
    cw = 256
    ch = 256
    h,w = image.shape[:2]
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi = gray[row:row+ch,col:col+cw]
            dst = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,20)
            gray[row:row+ch,col:col+cw] = dst
            print(np.std(dst),np.mean(dst))
            """
            全局二值化结合方差过滤实现类似自适应二值化的效果
            dev = np.std(roi)
            if dev<15:
                gray[row:row+ch,col:col+cw] = 255
            else:
                ret,dst = cv.threshold(roi,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
                gray[row:row+ch,col:col+cw] = dst
            """
    cv.imshow("result",gray)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
large_image_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()