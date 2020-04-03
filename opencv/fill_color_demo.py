import cv2.cv2 as cv
import numpy as np

def fill_color_demo(image):#颜色填充
    copyImg = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)
    cv.floodFill(copyImg,mask,(0,0),(255,255,0),(100,100,100),(100,100,30),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("rst",copyImg)

def fill_binary():#mask的指定的位置为零时才填充，不为零不填充
    image = np.zeros([400,400,3],np.uint8)
    image[100:300,100:300,:] = 255
    cv.imshow("fill banary",image)

    mask = np.ones([402,402,1],np.uint8)
    mask[101:301,101:301] = 0
    cv.floodFill(image,mask,(200,200),(100,2,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled banary",image)

src = cv.imread("G:/openCV/opencv/sources/samples/data/WindowsLogo.jpg")
fill_binary()

cv.waitKey(0)
cv.destroyAllWindows()