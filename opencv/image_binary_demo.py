import cv2.cv2 as cv
import numpy as np

def threshold_demo(image):#固定阈值二值化处理
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #THRESH_OTSU最大类间方差法，THRESH_THRIANGLE三角阈值法
    #THRESH_BINARY_INC二值取反，THRESH_TRUNC截断法（以阈值为上界），THRESH_TOZERO（以阈值为下界）
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    #手动选定阈值
    #ret,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    print("threshold value %s"%ret)
    cv.imshow("result",binary)

def local_threshold_demo(image):#自适应阈值二值化处理
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)#blocksize必须为奇数
    cv.imshow("binary",binary)

def custom_threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,h*w])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret,binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY)
    cv.imshow("binary",binary)


    
src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
custom_threshold_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()