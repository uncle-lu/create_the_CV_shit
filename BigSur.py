# created by uncle-lu
# final CheckSystem for Shit
# 2021.05.24

import os
import sys
import cv2
import numpy as np
import Catalina as Catalina
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
pth_dir = "./save_pth"
dataset_name = "road_train"
dataset_json_path = "./out.json"
dataset_dir_path = "./final_dataset"

def get_dicts():
    register_coco_instances(dataset_name, {}, dataset_json_path, dataset_dir_path)
    MetadataCatalog.get(dataset_name).set(thing_classes=['LeftAndStraight', 'LeftStraightRight', 'RightAndStraight', 'left', 'right', 'straight'])

def init_cfg(is_check):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.DATASETS.TRAIN = ("road_train", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 3
    cfg.OUTPUT_DIR = pth_dir
    if is_check:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.RETINANET.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

def imshow(name, img):
    while True:
        cv2.imshow(name, img)
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('ESC'):
            break
    cv2.destroyAllWindows()

def point1_maker(arg):
    left = 0
    right = 0
    straight = 0
    for i in arg:
        if i == 0 or i == 1 or i == 3:
            left = 1
        if i == 1 or i == 2 or i == 4:
            right = 1
        if i == 0 or i == 1 or i == 2 or i == 5:
            straight = 1

    ans = str(left) + str(straight) + str(right)
    return ans

def check_img(img, road_metadata):
    predictor = DefaultPredictor(cfg)
    output = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                   metadata=road_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW)
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1], output

def check_file(path, road_metadata):
    input = cv2.imread(path)
    img , output = check_img(input, road_metadata)
    print(point1_maker(output["instances"].pred_classes))
    imshow(path, img)

def check_folder(path, road_metadata):
    g = os.walk(path)
    result_dir = os.path.join(path, "result")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for path,dir_list,file_list in g:
        for file_name in file_list:
            if file_name[-3:] != "jpg":
                continue
            input = cv2.imread(os.path.join(path, file_name))
            img, output = check_img(input, road_metadata)
            result_file = open(os.path.join(result_dir, "{0:04d}.txt".format(int(file_name[:-4]))), "w")
            result_file.write(str(point1_maker(output["instances"].pred_classes)) + "\n")
            result_file.close()
            print("finish {0:04d}.txt".format(int(file_name[:-4])))

def old_main():
    get_dicts()

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == 'file':
            init_cfg(True)
            road_metadata = MetadataCatalog.get(dataset_name)
            file_path = str(sys.argv[2])
            check_file(file_path, road_metadata)
        elif str(sys.argv[1]) == 'folder':
            init_cfg(True)
            road_metadata = MetadataCatalog.get(dataset_name)
            folder_path = str(sys.argv[2])
            check_folder(folder_path, road_metadata)
        else:
            print("yead")
            init_cfg(True)
            road_metadata = MetadataCatalog.get(dataset_name)
            folder_path = "./test"
            check_folder(folder_path, road_metadata)

if __name__ == "__main__":
    get_dicts()
    init_cfg(True)
    road_metadata = MetadataCatalog.get(dataset_name)
    folder_path = "./data"
    check_folder(folder_path, road_metadata)