import cv2.cv2 as cv
import numpy as np
"""
            CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
　　　　　 　CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
　　　　　 　CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
　　　　　 　CV_TM_SQDIFF_NORMED 归一化平方差匹配法
　　　　　 　CV_TM_CCORR_NORMED 归一化相关匹配法
　　　　　 　CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
"""
def template_demo():
    tpl = cv.imread("C:/Users/32936/Desktop/2/eye.png")
    target = cv.imread("C:/Users/32936/Desktop/2/lena.png")
    cv.imshow("tpl",tpl)
    cv.imshow("target",target)
    methods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]#方差,相关性，相关性因子
    th,tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target,tpl,md)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:#方差越小越好
            tl = min_loc
        else:#相关性越大越好
            tl = max_loc
        br = (tl[0]+tw,tl[1]+th)
        cv.rectangle(target,tl,br,(0,0,255),2)#tl是位置，br是终止位置
        #cv.imshow("match"+np.str(md),target)
        cv.imshow("result"+np.str(md),result)
template_demo()
cv.waitKey(0)
cv.destroyAllWindows()