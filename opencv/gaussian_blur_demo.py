import cv2.cv2 as cv 
import numpy as np

def clamp(pv):
    if pv>255:
        return 255
    elif pv<0:
        return 0
    return pv

def gaussian_noise(image):#为图像加上高斯噪声
    h,w = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            image[row,col,0] = clamp(b + s[0])
            image[row,col,1] = clamp(g + s[1])
            image[row,col,2] = clamp(r + s[2])
    cv.imshow("image",image)
    return image

src = cv.imread("C:/Users/32936/Desktop/2/1.jpg")
t1 = cv.getTickCount()
rst = cv.GaussianBlur(gaussian_noise(src),(0,0),15)
t2 = cv.getTickCount()
t = (t2 - t1)/cv.getTickFrequency()
print("spend time:%s"%(t))
cv.imshow("result",rst)
cv.waitKey(0)
cv.destroyAllWindows()