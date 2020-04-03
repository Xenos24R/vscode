import cv2.cv2 as cv
import numpy as np

def top_hat_demo(image):#顶帽法=原始图像-开运算图像，得到噪声图像
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel)
    cimage = np.array(gray.shape,np.uint8)
    cimage = 100
    dst = cv.add(dst,cimage)
    cv.imshow("result",dst)


def black_hat_demo(image):#黑帽法=闭运算图像-原始图像，得到图像内部小孔或景色中的小黑点
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    dst = cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel)
    cimage = np.array(gray.shape,np.uint8)
    cimage = 100
    dst = cv.add(dst,cimage)
    cv.imshow("result",dst)

def gradient1_demo(image):#基本梯度求取,可提取图像边缘
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst = cv.morphologyEx(binary,cv.MORPH_GRADIENT,kernel)
    cv.imshow("dst",dst)

def gradient2_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dm = cv.dilate(image,kernel)
    ed = cv.erode(image,kernel)
    dst1 = cv.subtract(dm,ed)
    dst2 = cv.subtract(ed,dm)
    cv.imshow("dst1",dst1)
    cv.imshow("dst2",dst2)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
gradient2_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()