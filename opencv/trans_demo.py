import cv2.cv2 as cv

def color_space_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)#cv.cvtColor():转换图片颜色空间
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    yuv = cv.cvtColor(image,cv.COLOR_BGR2YUV)
    Ycrcb = cv.cvtColor(image,cv.COLOR_BGR2YCrCb)
    cv.imshow("grag",gray)
    cv.imshow("hsv",hsv)
    cv.imshow("yuv",yuv)
    cv.imshow("Ycrcb",Ycrcb)
src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
color_space_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()