import numpy as np
import cv2.cv2 as cv
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import random
#from google.colab.patches import cv2_imshow

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
#下载图片
#wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv.imread("C:/Users/32936/Desktop/OpenCV/party.jpg")
capture = cv.VideoCapture(0)

cfg = get_cfg()
cfg.merge_from_file("E:/object_detection/detectron2-master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
#cfg.MODEL.WEIGHTS = "./COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)

while(True):
    ret,frame = capture.read()
    frame = cv.flip(frame,1)
    if ret==False:
        break
    outputs = predictor(im)
    #outputs = predictor(frame)
    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes
    #在原图上画出检测结果
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv.imshow("image",v.get_image())
    c = cv.waitKey(40)
    if c == 27:
        break

#plt.imshow(v.get_image())
#plt.show()
