import os
import numpy as np
import cv2
import io
import json
from base64 import b64decode,b64encode
from flask import request
from PIL import Image
from glob import glob
import time as tick
class FileControl:
    """
    media관련 path를 return해주는 함수들을 갖고 있는 class
    """
    def __init__(self):
        this_dir = os.path.abspath(os.path.dirname(__file__))
        base_dir = os.path.abspath(os.path.dirname(this_dir))

        media_dir = os.path.join(base_dir, 'media')
        if not os.path.exists(os.path.join(media_dir, 'image')):
            os.makedirs(os.path.join(media_dir, 'image'))
        self.image_dir = os.path.join(media_dir, 'image')
        self.video_dir = os.path.join(media_dir, 'video')
        self.tracking_dir = os.path.join(media_dir,'tracking')

    def get_image_path(self, location, time, return_join=True):
        pre_path = os.path.join(self.image_dir,f'{location}')
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
        image_path = os.path.join(pre_path ,time + '.jpg')
        if return_join:
            return os.path.join(self.image_dir, image_path)
        else:
            return self.image_dir, image_path

    def get_video_path(self, video_name):
        video_path = os.path.join(self.video_dir, video_name)
        return video_path
    
    def get_tracked_image_path(self,location,time,return_join=True):
        image_path = os.path.join(f'{location}',time+'.jpg')
        if not return_join:
            return self.tracking_dir,image_path        
        return  os.path.join(self.tracking_dir,image_path)


def get_param_parsing():
    """Get_parsing

    Returns:
        location ([int]): 요청된 location
        time ([str]): 요청된 이미지에 대한 시간 정보
    """
    args = request.args
    param_name = ['location', 'time', 'onlyimage']
    param_default = [0, '0000_00_00_00_00_00_00', False,False]
    param_type = [int, str, bool]
    params = []
    for name, default, data_type in zip(param_name, param_default, param_type):
        params.append(args.get(name, default=default, type=data_type))

    return params

def get_location_parsing():
    """Get_parsing

    Returns:
        location ([int]): 요청된 location
    """
    args = request.args
    return args.get('location', default=0, type=int)

def get_frame_parsing():
    """Get_parsing

    Returns:
        location ([int]): 요청된 location
    """
    args = request.args
    return args.get('frame', default=1, type=int)

def get_time_list(location):
    """ 특정 location에 대한 time_list 반환
    Args:
        location ([int]): 요청된 location

    Returns:
        ([str list]): 특정 location에 대한 time_list
    """
    image_paths = glob(f'./media/tracking/{location}/*.jpg')
    return [{'time_info': image_path.split('\\')[-1].replace('.jpg','')} for image_path in image_paths]

def get_box_num(location,time):
    """ 특정 location, time에 대해 검출된 box 개수 반환

    Args:
        location ([int]): 요청된 장소
        time ([str]): 요청된 시간

    Returns:
        [int]: 검출된 box 개수
    """
    path = os.path.join('./media/tracking',f'{location}',f'box_{time[:10]}.csv')
    lines = open(path,'r')
    for line in lines:
        l = line.split(',')
        if l[0] == time:
            return int(l[1].strip('\n'))
    return -1

def get_image():
    """Get_parsing

    Returns:
        ([np.array]): 디코딩 된 이미지
    """
    image = request.files['file'].read()
    npimg = np.frombuffer(b64decode(image),dtype=np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)

def load_encoded_img(img_path):
    """[summary]

    Args:
        img_path ([str]): 이미지 경로

    Returns:
        [base64 object]: 인코딩 된 이미지
    """
    img_file = open(img_path, "rb")
    encoded_img = b64encode(img_file.read()).decode('ascii')
    img_file.close()
    return encoded_img

def encode_img(img):
    """Tracking 결과 이미지 인코딩

    Args:
        img (np.array): Tracking 결과 이미지

    Returns:
        [base64 object]: 인코딩 된 이미지
    """
    _, encoded_img = cv2.imencode('.jpg', img)
    return b64encode(encoded_img).decode('ascii')


if __name__ == '__main__':
    # media/video/box.mp4 -> frames save to media/image/
    import datetime as dt
    now_time = dt.datetime.now()

    video_name = 'box.mp4'
    path = FileControl()
    video_path = path.get_video_path(video_name)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    location = 0
    count = 0
    while success:
        time = now_time+ dt.timedelta(seconds=0.033*count)
        time = time.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-4]
        image_path = path.get_image_path(location, time)
        cv2.imwrite(image_path, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame #{}: {}'.format(count, success))
        count += 1
