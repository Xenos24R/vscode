import cv2.cv2 as cv
import numpy as np

def gauss_pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_image = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_image.append(dst)
        cv.imshow("pyramid"+np.str(i),dst)
        temp = dst.copy()
    return pyramid_image
    
def laplace_pyramid_demo(image):#必须宽高相等
    pyramid_image = gauss_pyramid_demo(image)
    level = len(pyramid_image)
    for i in range(level-1,-1,-1):
        if (i-1)<0:
            expand = cv.pyrUp(pyramid_image[i],dstsize=image.shape[:2])
            lpls = cv.subtract(image,expand)
            cv.imshow("laplace"+str(i),lpls)
        else:
            expand = cv.pyrUp(pyramid_image[i],dstsize=pyramid_image[i-1].shape[:2])
            lpls = cv.subtract(pyramid_image[i-1],expand)
            cv.imshow("laplace"+str(i),lpls)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
laplace_pyramid_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()