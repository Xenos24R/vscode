import cv2.cv2 as cv
import numpy as np

cap = cv.imread("C:/Users/32936/Desktop/2/cap.jpg")
model = cv.imread("C:/Users/32936/Desktop/2/book.jpg")

cap_slave = cap
MIN_MATCHES = 15

#启动ORB探测器
orb = cv.ORB_create()

#matcher对象
bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)

#计算模型关键点及描述符
kp_model,des_model = orb.detectAndCompute(model,None)

#计算场景关键点及描述符
kp_frame,des_frame = orb.detectAndCompute(cap,None)

#匹配帧描述符和模型描述符
matches = bf.match(des_model,des_frame)

#按距离排序
matches = sorted(matches,key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    #cap = cv.drawMatches(model,kp_model,cap,kp_frame,matches[:MIN_MATCHES],0,flags=2)

    cv.imshow('frame',cap)
else :
    print("not enough matches have been found - %d/%d" % (len(matches),MIN_MATCHES))

src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

M,mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC,5.0)

h,w = model.shape[0:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

dst = cv.perspectiveTransform(pts,M)
rst = cv.warpPerspective(cap,M,(cap.shape[1],cap.shape[0]))

#img2 = cv.polylines(cap_slave,[np.int32(dst)],True,255,3,cv.LINE_AA)
cv.imshow('frame',rst)
cv.imshow("image",cap)
cv.waitKey()