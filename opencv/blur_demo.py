import cv2.cv2 as cv 
import numpy as np

def blur_demo(image):#均值模糊,去噪
    dst = cv.blur(image,(1,20))#第二个参数是卷积范围
    cv.imshow("image",dst)

def middle_blue_demo(image):#中值模糊，去除椒盐噪声
    dst = cv.medianBlur(image,5)
    cv.imshow("image",dst)

def custom_blur_demo(image):#自定义滤波，模板总和为1做增强，模板总和为0做边缘梯度
    #kernel = np.ones([5,5],np.float32)/25#均值滤波
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)/9#锐化算子
    dst = cv.filter2D(image,-1,kernel)
    cv.imshow("image",dst)

src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
custom_blur_demo(src)
cv.waitKey(0)