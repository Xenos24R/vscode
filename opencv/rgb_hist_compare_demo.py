import cv2.cv2 as cv
import numpy as np

def creat_rgb_hist(image):
    h,w = image.shape
    rgbHist = np.zeros([16*16*16,1],np.float32)#降维
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            index = (b//bsize)*16*16 + (g//bsize)*16 + (r//bsize)#将三维数组转化为一维
            rgbHist[np.int(index),0] += 1
    return rgbHist

def hist_compare(image1,image2):
    hist1 = creat_rgb_hist(image1)
    hist2 = creat_rgb_hist(image2)
    match1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)#巴氏距离，越小越相似
    match2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)#相关性比较，越大越相似
    match3 = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)#卡方，越大越不相似
    print("巴氏：%s,相关性：%s,卡方：%s"%(match1,match2,match3))

src1 = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
src2 = cv.imread("C:/Users/32936/Desktop/2/1.jpg")

hist_compare(src1,src2)