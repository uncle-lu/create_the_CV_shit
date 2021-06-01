# created by uncle-lu
# the mini train for test
# 2021.05.21

import cv2
import random
import os
import json
import numpy as np
import detectron2
import sys
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

setup_logger()
cfg = get_cfg()

def get_dicts():
    register_coco_instances("road_train", {}, "./final_val.json", "./final_dataset")
    MetadataCatalog.get("road_train").set(thing_classes=['LeftAndStraight', 'LeftStraightRight', 'RightAndStraight', 'left', 'right', 'straight'])

def test_img(path, road_metadata):
    img = cv2.imread(path)
    visualizer = Visualizer(img[:, :, ::-1], metadata=road_metadata, scale=0.5,instance_mode=ColorMode.IMAGE_BW)
    t = {}
    for i in DatasetCatalog.get("road_train"):
        if i["file_name"] == path:
            t = i
    vis = visualizer.draw_dataset_dict(t)
    while True:
        cv2.imshow("img", vis.get_image()[:, :, ::-1])  
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def init_cfg():
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.DATASETS.TRAIN = ("road_train", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 3
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

def train_test():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load()
    trainer.train()

def check(path, road_metadata):
    predictor = DefaultPredictor(cfg)

    im = cv2.imread(path)
    outputs = predictor(im)
    print(outputs)
    v = Visualizer(im[:, :, ::-1],
            metadata=road_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW
            )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    while True:
        cv2.imshow('img',img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_dicts()
    init_cfg()
    road_metadata = MetadataCatalog.get("road_train")

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == 'test':
            test_img("./final_dataset/498.jpg", road_metadata)
        elif str(sys.argv[1]) == 'train':
            train_test()
        elif str(sys.argv[1]) == 'check':
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            check("./main.jpg", road_metadata)
