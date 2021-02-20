from PIL import Image
from flask import Flask, make_response, send_from_directory
from flask_restful import Resource, Api
from utils.utils import get_location_parsing, get_param_parsing, get_frame_parsing, get_image, encode_img,\
    FileControl, get_box_num, get_time_list, load_encoded_img

from utils.recognition_utils import Tracker
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

path = FileControl()

UPLOAD_FOLDER = path.image_dir
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


class LoadConfig:
    def __init__(self, opt, args, arc_model, face_detector, person_detector, deepsort, Faces, tracked):
        self.opt = opt
        self.args = args
        self.arc_model = arc_model
        self.face_detector = face_detector
        self.person_detector = person_detector
        self.deepsort = deepsort
        self.Faces = Faces
        self.tracked = tracked
        self.out = None
        if args.write:
            fps = 30
            fcc = cv2.VideoWriter_fourcc(*'FMP4')  # cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            self.out = cv2.VideoWriter(f'{args.result_name}.mp4', fcc, fps, (args.write_size, args.write_size))
        self.img = None

    def set_img(self, img):
        self.img = img

    def get_config(self):
        return self.opt, self.args, self.arc_model, self.face_detector, self.person_detector,\
               self.deepsort, self.Faces, self.tracked, self.out, self.img

class Index(Resource):
    description = '<h1>API Description</h1>'\
                  '<table border="1" style="border-collapse:collapse">'\
                  '<tr><td>/index</td><td>GET</td><td>API 설명 페이지(this)</td></tr>'\
                  '<tr><td>/tracking</td><td>POST</td><td>image를 전송해 tracking 처리하여 저장하는 API</td></tr>'\
                  '<tr><td>/get_image</td><td>GET</td><td>Image를 요청하여 받는 API</td></tr>'\
                  '<tr><td>/room_info</td><td>GET</td><td>카메라 번호(Room 번호) 리스트를 요청하여 받는 API</td></tr>'\
                  '</table>'

    def get(self):
        res = make_response(self.description)
        res.headers['Content-type'] = 'text/html; charset=utf-8'
        return res

class ReleaseOut(Resource):
    def get(self):
        t0 = time.time()
        tracking.remove_out()
        res = make_response({'Status': 'Success', 'process_time': time.time() - t0})
        res.headers['Content-type'] = 'text/html; charset=utf-8'
        return res

class TrackingImage(Resource):
    """Image를 HTTP post 방식으로 받아서 tracking해주는 API
        location(방 번호)와 frame(프레임 번호)는 GET 방식으로 입력받음
    
    Args:
        Resource ([Resource object]): /tracking?location=a&time=b
            a ([int]): 트레킹 요청된 location, default = 0
            b ([str]): 트레킹 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
        Request file ([base64 object]): 트레킹 요청된 이미지
    
    Returns:
        ([Response object])
            file_path ([str]): 이미지 저장된 경로,
            image ([base64 object]): Tracking 결과 이미지
            box_num ([int]):검출된 box 개수
    """
    # TODO: 매 프레임마다 해야하는 동작에 return이 불필요하고 많음

    def get(self):
        return self.post()

    @staticmethod
    def post():
        t0 = time.time()
        # location, time_, _ = get_param_parsing()
        frame_num = get_frame_parsing()
        print(f'Parsing time : {time.time() - t0}')

        img = get_image()
        t1 = time.time()
        print(f'Decoding time : {time.time() - t1}')

        t2 = time.time()
        # file_path, img, box_num = tracking.face_track_img(img)
        img = tracking.face_track_img(img, frame_num)
        # encoded_img = encode_img(img)
        print(f'inference time : {time.time() - t2}')
    
        res = make_response({})
        # res = make_response({'image': encoded_img})
        res.headers['Content-type'] = 'application/json'

        return res


class SendImage(Resource):
    """특정 location, time에 해당하는 결과 이미지, 검출된 박스 개수 반환
    
    Args:
        Resource ([Resource object]): /get_image?location=a&time=b
            a ([int]): 요청된 location, default = 0
            b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
    Returns:
        Json
        ([Response object])
            image ([base64 object]): Tracking 결과 이미지
            box_num ([int]):검출된 box 개수
    """
    
    @staticmethod
    def get():
        location, time, show_image = get_param_parsing()
        
        # return time, saved_image, box num
        if show_image:
            file_path, file_name = path.get_tracked_image_path(location, time, return_join=False)

            return send_from_directory(file_path, file_name)
        else:
            img_path = path.get_tracked_image_path(location, time)
            encoded_img = load_encoded_img(img_path)
            
            box_num = get_box_num(location,time)
            res = make_response({'image': encoded_img, 'box_num': box_num})
            res.headers['Content-type'] = 'application/json'

            return res

class SendInfo(Resource):
    """특정 location에 대한 time_list 반환
    
    Args:
        Resource ([Resource object]): /location_info?location=a
            a = 요청된 location, default = 0
    Returns:
        ([str list]): 특정 location에 대한 time_list
    """
    @staticmethod
    def get():
        location = get_location_parsing()
        res = make_response({'time_list': get_time_list(location)})
        res.headers['Content-type'] = 'application/json'
        return res

class SendRooms(Resource):
    """특정 location에 대한 time_list 반환

    Args:
        Resource ([Resource object]): /location_info?location=a
            a = 요청된 location, default = 0
    Returns:
        ([str list]): 특정 location에 대한 time_list
    """

    @staticmethod
    def get():
        import glob
        room_list = glob.glob('./media/tracking/*')
        room_list = [room.split('/')[-1] for room in room_list]
        if len(room_list[0].split('\\')) > 1:
            room_list = [room.split('\\')[-1] for room in room_list]
        room_list = [{'room_num': room_num} for room_num in room_list]
        res = make_response({'room_list': room_list})
        res.headers['Content-type'] = 'application/json'
        return res


api.add_resource(Index, '/', '/index')
# API를 간단히 설명해주는 페이지

api.add_resource(ReleaseOut, '/releaseout')
# VideoWriter를 메모리에서 반환해주기 위한 API

api.add_resource(TrackingImage, '/tracking')
# /tracking?location=a&time=b
# a ([int]): 요청된 location, default = 0
# b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'
# 

api.add_resource(SendImage, '/get_image')
# /get_image?location=a&time=b
# a ([int]): 요청된 location, default = 0
# b ([str]): 요청된 이미지에 대한 시간 정보, default = '0000_00_00_00_00_00_00'

api.add_resource(SendInfo, '/location_info')
# /location_info?location=a
# a = 방의 위치(카메라의 번호), default=0

api.add_resource(SendRooms, '/room_info')
# /room_info

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Arc face & Retina face inference')
    # input
    parser.add_argument('--src_path', type=str, help='Path of input data', default='')
    parser.add_argument('--is_resize', action='store_true',
                        help='Default is false, If you want to resize input image, use --is_resize')
    parser.add_argument('--resize', type=int, default=416, help='Size of resized image')

    # save
    parser.add_argument('--write', action='store_true', help='Default is false,If you want to save result, use --write')
    parser.add_argument('--write_size', type=int, default=416, help='Image or Frame size of result')
    parser.add_argument('--result_name', type=str, help='Name of result file. Please set --write', default='result')

    # threshold
    parser.add_argument('--indivisual_threshold', action='store_true',
                        help='Default is false, If you use this flag, confidence scroe and similarity threshold will be applied indivisually')

    # util
    parser.add_argument('--is_draw', action='store_true', help='Default is false, If you want to draw img')
    args = parser.parse_args()

    opt = Config()
    arc_model = resnet_face18(opt.use_se)
    load_model(arc_model, opt.arc_model_path)

    cudnn.benchmark = True
    device = torch.device("cuda")

    # load arcface model
    arc_model.to(device)
    arc_model.eval()

    # load retina face model (face detector)
    face_detector = RetinafaceDetector(is_gpu=False, opt=opt)

    # load deep sort model
    deepsort = DeepSort(opt.deepsort_cfg.DEEPSORT.REID_CKPT,
                        max_dist=opt.deepsort_cfg.DEEPSORT.MAX_DIST,
                        min_confidence=opt.deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=opt.deepsort_cfg.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=opt.deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=opt.deepsort_cfg.DEEPSORT.MAX_AGE, n_init=opt.deepsort_cfg.DEEPSORT.N_INIT,
                        nn_budget=opt.deepsort_cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # load yolo5 model (person detector)
    person_detector = torch.load(opt.yolo_model_path, map_location=device)['model'].float().half()

    # load face featrues
    Faces = FaceFeatures(opt.features_path)

    tracked = TrackFace(max_cnt=5, detect_threshold=opt.detect_threshold, sim_threshold=opt.sim_threshold)

    tracking = Tracker(opt, args, arc_model, face_detector, person_detector, deepsort, Faces, tracked)

    app.run(host='0.0.0.0')
    # app.run()
