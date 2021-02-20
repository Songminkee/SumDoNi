import sys

sys.path.insert(0, './yolov5')

from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized
from utils.parser import get_config
from deep_sort import DeepSort

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from config import Config
from arcface.models import resnet_face18
from retinaface.detector import RetinafaceDetector
from utils.face_util import *
from utils.detect_util import *


class Tracker:
    def __init__(self, opt, args, arc_model, face_detector, person_detector, deepsort, Faces, tracked):
        self.config = opt
        self.args = args
        self.arc_model = arc_model
        self.face_detector = face_detector
        self.person_detector = person_detector
        self.deep_sort = deepsort
        self.Faces = Faces
        self.tracked = tracked
        self.out = None

    def create_out(self):
        args = self.args
        if args.write:
            fps = 30
            fcc = cv2.VideoWriter_fourcc(*'FMP4')  # cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            self.out = cv2.VideoWriter(f'{args.result_name}.mp4', fcc, fps, (args.write_size, args.write_size))

    def remove_out(self):
        self.out.release()
        self.out = None

    def face_track_img(self, img_raw, frame):
        tic = time.time()
        if self.args.is_resize:
            img_raw = cv2.resize(img_raw, (self.args.resize, self.args.resize), interpolation=cv2.INTER_LINEAR)

        img = torch.from_numpy(np.float32(img_raw.copy()).transpose(2, 0, 1)).to('cuda').half()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print(f"pre", time.time() - tic)
        t2 = time.time()
        pred = self.person_detector(img, augment=False)[0]

        det = non_max_suppression(
            pred, self.config.yolo_confidence_threshold, self.config.yolo_nms_threshold, classes=[0], agnostic=False)[0]
        print(f"detection,nms time = {time.time() - t2}")
        if len(det):
            t3 = time.time()
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img.shape[2:]).round()

            bbox_xywh = []
            confs = []

            for *xyxy, conf, cls in det:
                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf.item()])

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            # Pass detections to deepsort
            print(f"deepsort preprocessing time = {time.time() - t3}")
            t4 = time.time()
            outputs = self.deep_sort.update(xywhs, confss, img_raw)
            print(f"deepsort time = {time.time() - t4}")
            print(outputs)
            if len(outputs):
                t5 = time.time()
                identities, bbox_xyxy = self.tracked.check_identities(outputs[:, :4], outputs[:, -1], frame)
                print(f"check_id time = {time.time() - t5}")
                if len(bbox_xyxy):
                    t6 = time.time()
                    face_boxes, face_det_scores, sim_scores, \
                    features, patches = face_recognition_multi(img_raw,
                                                               self.arc_model,
                                                               self.face_detector,
                                                               self.Faces,
                                                               # draw_img=self.args.is_draw,
                                                               indivisual_threshold=self.args.indivisual_threshold,
                                                               bbox_xyxy=bbox_xyxy)
                    print(f"face detect,recog time = {time.time() - t6}")
                    t7 = time.time()
                    self.tracked.update(self.Faces, identities, bbox_xyxy, face_boxes, face_det_scores,
                                        sim_scores, features, patches, str(frame))
                    print(f"update time = {time.time() - t7}")

        t9 = time.time()
        self.tracked.age_update()
        print(f"age time = {time.time() - t9}")

        img_draw = img_raw.copy()
        if self.args.is_draw:
            t8 = time.time()
            self.tracked.draw_info(img_draw)
            print(f"draw time = {time.time() - t8}")
            cv2.putText(img_draw, f'now_frame={frame}', (0, 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.imshow('result', img_draw)
            k = cv2.waitKey(1)
            if k == ord('s'):
                make_feature(self.config, self.Faces, self.arc_model, self.face_detector, img_raw)

        if self.args.write:
            if not self.out:
                self.create_out()

            self.out.write(cv2.resize(img_draw, (self.args.write_size, self.args.write_size), interpolation=cv2.INTER_LINEAR))
            img_raw = img_draw

        print('tracked', self.tracked)
        print(f"time = {time.time() - tic}\n")

        return img_raw

#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Arc face & Retina face inference')
#     # input
#     parser.add_argument('--src_path', type=str, help='Path of input data', default='')
#     parser.add_argument('--is_resize', action='store_true',
#                         help='Default is false, If you want to resize input image, use --is_resize')
#     parser.add_argument('--resize', type=int, default=416, help='Size of resized image')
#
#     # save
#     parser.add_argument('--write', action='store_true', help='Default is false,If you want to save result, use --write')
#     parser.add_argument('--write_size', type=int, default=416, help='Image or Frame size of result')
#     parser.add_argument('--result_name', type=str, help='Name of result file. Please set --write', default='result')
#
#     # threshold
#     parser.add_argument('--indivisual_threshold', action='store_true',
#                         help='Default is false, If you use this flag, confidence scroe and similarity threshold will be applied indivisually')
#
#     # util
#     parser.add_argument('--is_draw', action='store_true', help='Default is false, If you want to draw img')
#     args = parser.parse_args()
#
#     opt = Config()
#     arc_model = resnet_face18(opt.use_se)
#     load_model(arc_model, opt.arc_model_path)
#
#     cudnn.benchmark = True
#     device = torch.device("cuda")
#
#     # load arcface model
#     arc_model.to(device)
#     arc_model.eval()
#
#     # load retina face model (face detector)
#     face_detector = RetinafaceDetector(is_gpu=False, opt=opt)
#
#     # load deep sort model
#     deepsort = DeepSort(opt.deepsort_cfg.DEEPSORT.REID_CKPT,
#                         max_dist=opt.deepsort_cfg.DEEPSORT.MAX_DIST,
#                         min_confidence=opt.deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
#                         nms_max_overlap=opt.deepsort_cfg.DEEPSORT.NMS_MAX_OVERLAP,
#                         max_iou_distance=opt.deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                         max_age=opt.deepsort_cfg.DEEPSORT.MAX_AGE, n_init=opt.deepsort_cfg.DEEPSORT.N_INIT,
#                         nn_budget=opt.deepsort_cfg.DEEPSORT.NN_BUDGET,
#                         use_cuda=True)
#
#     # load yolo5 model (person detector)
#     person_detector = torch.load(opt.yolo_model_path, map_location=device)['model'].float().half()
#
#     # load face featrues
#     Faces = FaceFeatures(opt.features_path)
#
#     face_track(opt, args, arc_model, face_detector, person_detector, deepsort, Faces)