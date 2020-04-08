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

src = cv.imread("C:/Users/32936/Desktop/2/lena.png")
custom_blur_demo(src)
cv.waitKey(0)


"""
均值滤波
cv2.blur(src, ksize[], dst[], anchor[], borderType)
    src:输入的图像
    ksize[]:模板大小
    anchor[]:锚点，处理的像素位于模板的什么位置
    borderType:用于推断图像外部像素的某种边界模式
    ->dst[]:与src大小相同的输出结果

中值滤波
cv2.medianBlur(src, ksize[], dst)
    src:输入的图像
    ksize[]:模板大小
    ->dst[]:与src大小相同的输出结果

卷积运算函数
cv2.filter2D(src, ddepth, kernel[], dst[], anchor[], delta[], borderType)
    src:输入的图像
    ddepth:目标图像深度，当ddepth输入值为-1时，目标图像和原图像深度保持一致。图像深度是指存储每个像素所用的位数，也用于量度图像的色彩分辨率
    kernel:卷积核
    anchor[]:锚点，处理的像素位于模板的什么位置
    delta[]:在储存目标图像前可选的添加到像素的值，默认值为0
    borderType:用于推断图像外部像素的某种边界模式
    ->dst[]:与src大小相同的输出结果
"""