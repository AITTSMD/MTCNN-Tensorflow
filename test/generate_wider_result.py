# coding:utf-8
import sys
import numpy as np

sys.path.append("..")
import argparse
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
import cv2
import os



def read_gt_bbox(raw_list):
    list_len = len(raw_list)
    bbox_num = (list_len - 1) // 4
    idx = 1
    bboxes = np.zeros((bbox_num, 4), dtype=int)
    for i in range(4):
        for j in range(bbox_num):
            bboxes[j][i] = int(raw_list[idx])
            idx += 1
    return bboxes


def get_image_info(anno_file):
    f = open(anno_file, 'r')
    image_info = []
    for line in f:
        ct_list = line.strip().split(' ')
        path = ct_list[0]

        path_list = path.split('\\')
        event = path_list[0]
        name = path_list[1]
        # print(event, name )
        bboxes = read_gt_bbox(ct_list)
        image_info.append([event, name, bboxes])
    print('total number of images in validation set: ', len(image_info))
    return image_info

if __name__ == '__main__':

    data_dir = '../../DATA/WIDER_val/images'
    anno_file = 'wider_face_val.txt'
    output_file = '../../DATA/WIDER_NoLM_RNet_0.3_0.1/'

    test_mode = "RNet"
    thresh = [0.3, 0.1, 0.7]
    min_face_size = 20
    stride = 2
    slide_window = False
    shuffle = False
    vis = False
    detectors = [None, None, None]
    # prefix is the model path
    prefix = ['../data/MTCNN_model/PNet_No_Landmark/PNet', '../data/MTCNN_model/RNet_No_Landmark/RNet',
              '../data/MTCNN_model/ONet_No_Landmark/ONet']
    epoch = [30, 14, 16]
    batch_size = [2048, 256, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    image_info = get_image_info(anno_file)

    current_event = ''
    save_path = ''
    idx = 0
    for item in image_info:
        idx+=1
        image_file_name = os.path.join(data_dir, item[0], item[1])
        if current_event != item[0]:

            current_event = item[0]
            save_path = os.path.join(output_file, item[0])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('current path:', current_event)

        # generate detection
        img = cv2.imread(image_file_name)
        all_boxes, _ = mtcnn_detector.detect_single_image(img)


        f_name = item[1].split('.jpg')[0]

        dets_file_name = os.path.join(save_path, f_name + '.txt')
        fid =open (dets_file_name,'w')
        boxes = all_boxes[0]
        if boxes is None:
            fid.write(item[1] + '\n')
            fid.write(str(1) + '\n')
            fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
            continue
        fid.write(item[1] + '\n')
        fid.write(str(len(boxes)) + '\n')

        for box in boxes:
            fid.write('%f %f %f %f %f\n' % (
            float(box[0]), float(box[1]), float(box[2] - box[0] + 1), float(box[3] - box[1] + 1), box[4]))

        fid.close()
        if idx % 10 == 0:
            print(idx)