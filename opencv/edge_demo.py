import cv2.cv2 as cv
import numpy as np

def sobel_demo(image):#当边缘不明显时可用Scharr算子，但该算子受噪声影响较大
    grad_x = cv.Sobel(image,cv.CV_32F,1,0)
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)

    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    cv.imshow("x_dir",gradx)
    cv.imshow("y_dir",grady)

    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("xy_dir",gradxy)

def laplace_demo(image):#拉普拉斯算子
    dst = cv.Laplacian(image,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("lpls",lpls)

def custom_demo(image):
    kernel = np.array([[1,1,1,],[1,-8,1],[1,1,1]])
    dst = cv.filter2D(image,cv.CV_32F,kernel=kernel)
    custom = cv.convertScaleAbs(dst)
    cv.imshow("result",custom)
    
src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
custom_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()