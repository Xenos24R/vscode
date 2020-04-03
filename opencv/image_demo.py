import cv2.cv2 as cv
import numpy as np

def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)

src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
get_image_info(src)
cv.imshow("input image",src)
cv.waitKey(0)
cv.destroyAllWindows()