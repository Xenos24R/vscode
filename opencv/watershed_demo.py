import cv2.cv2 as cv
import numpy as np

def watershed_demo(image):
    blurred = cv.pyrMeanShiftFiltering(image,10,100)
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow("binary",binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    mb = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=2)
    sure_bg = cv.dilate(binary,kernel,iterations=3)
    cv.imshow("mor",sure_bg)

    dist = cv.distanceTransform(mb,cv.DIST_L2,3)
    dist_output = cv.normalize(dist,0,1.0,cv.NORM_MINMAX)
    cv.imshow("dist",dist_output*50)

    ret,surface = cv.threshold(dist,dist.max()*0.6,255,cv.THRESH_BINARY)
    cv.imshow("interface",surface)

    surface_fg = np.uint8(surface)
    unknow = cv.subtract(sure_bg,surface_fg)
    ret,markers = cv.connectedComponents(surface_fg)
    print(ret)

    markers += 1
    markers[unknow==255] = 0 
    markers = cv.watershed(image,markers=markers)
    image[markers==-1] = [0,0,255]
    cv.imshow("result",image)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
watershed_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()