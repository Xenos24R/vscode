import cv2.cv2 as cv
import numpy as np

def contours_demo(image):
    dst = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,255,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#阈值二值化处理
    cv.imshow("binary image",binary)

    contours,heriachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    print(np.size(contours))
    for i,contour in enumerate(contours):
        cv.drawContours(image,contours,i,(0,0,255),2)#最后一个参数-1为填充
        print(i)
        print(contour)
    cv.imshow("result",image)

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
contours_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()


"""
全局阈值二值化
cv.threshold(src, thresh, maxval, type, dst)
    src:输入的图像
    thresh:阈值，当使用OTSU等自适应算法时会自动生产一个阈值
    maxval:高于/低于阈值时赋予的新值
    type:二值化的方法
    ->retval:阈值
    ->dst:与src大小相同的二值化图片

寻找图像轮廓
cv2.findContours(image, mode, method, contours[], hierarchy[], offset[])
    image:输入的二值图像
    mode:轮廓的检索模式
    method:轮廓的近似方法
    offset[]:轮廓的偏移值
    ->contours[]:图片中轮廓的坐标
    ->hierarchy[]:返回一个N x 4大小的矩阵，存放轮廓的相互关系，下有详解

cv2.drawContours(image, contours, contourIdx, color[], thickness, lineType, hierarchy[], maxLevel[], offset[])
    image:输入的图像，三通道才能显示轮廓
    contours:轮廓
    contourIdx:绘制轮廓列表中某一条轮廓的轮廓编号，输入-1则绘制所有的轮廓
    color:轮廓颜色
    thickness:轮廓的粗细
    lineType:轮廓线是四连通线或八连通线
    hierarchy[]:输出的轮廓层次，与maxlevel共同起作用
    maxLevel:限制轮廓的最高层次，输入0则只绘制最高层次的轮廓
    offset:轮廓的偏移值

非极大值抑制（NMS）主要是为了更精确的定位某种特征，比如用梯度变化表征边缘时，梯度变化较大的区域通常比较宽，所以利用
x和y方向的梯度确定一个法向arctan(y/x)，然后在法向上判断当前梯度测量是否是一个峰值（或局部极大值），如果是就保留，不
是极大值就抑制（如设置为0）。这样的话就能将边缘定位在1-2像素宽

hierarchy详解：
|-----------------------------------------------------------------------------------------------------------------|
|    第一个数：表示同一级轮廓的下个轮廓的编号，如果这一级轮廓没有下一个轮廓，一般是这一级轮廓的最后一个的时候，则为-1    |
|                                                                                                                 |
|    第二个数：表示同一级轮廓的上个轮廓的编号，如果这一级轮廓没有上一个轮廓，一般是这一级轮廓的第一个的时候，则为-1      |
|                                                                                                                 |
|    第三个数：表示该轮廓包含的下一级轮廓的第一个的编号，假如没有，则为-1                                             |
|                                                                                                                 |
|    第四个数： 表示该轮廓的上一级轮廓的编号，假如没有上一级，则为-1                                                  |
|-----------------------------------------------------------------------------------------------------------------|
"""