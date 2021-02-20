import requests
import cv2
import time
import datetime as dt
import base64

from flask import Flask, make_response, send_from_directory
from flask_restful import Api, Resource
from utils.utils import FileControl, get_location_parsing

app = Flask(__name__)
api = Api(app)
path = FileControl()


class Index(Resource):
    description = '<h1>API Description</h1>'\
                  '<table border="1" style="border-collapse:collapse">'\
                  '<tr><td>/index</td><td>GET</td><td>API 설명 페이지(this)</td></tr>'\
                  '<tr><td>/reqvideo</td><td>GET</td><td>Webcam, Video 등을 읽어 frame 하나씩 inference 요청하는 API</td></tr>'\
                  '</table>'

    def get(self):
        res = make_response(self.description)
        res.headers['Content-type'] = 'text/html; charset=utf-8'
        return res


class RequestVideo(Resource):
    # - IP주소: 121.151.100.57
    # - 포트: 9001
    ip = '211.55.92.51'
    port = 5000
    url = f'http://{ip}:{port}/tracking'
    source = {0: 'inputs/1.mp4',
              1: 'rtsp://hanamart-public:hanamart-public@121.151.100.57:9001//h264Preview_01_sub'}

    def get(self):
        location = get_location_parsing()
        # use_webCam = False
        # if use_webCam:
        #     source = 0
        # else:
        #     source = path.get_video_path('box.mp4')
        repeat_count = 0
        while repeat_count < 10:
            cap = cv2.VideoCapture(self.source[location])
            if not cap.isOpened():
                cv2.waitKey(1)
                repeat_count += 1
                print(f'Try load Video(#{repeat_count}) : {self.source[location]}')
                continue

            payload = {}
            headers = {}
            idx = 0
            start_point = 0
            now_time = dt.datetime.now()
            avg_reference_time = 0.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx >= start_point:
                    _, img_encoded = cv2.imencode('.jpg', frame)

                    files = [
                        ('file', (f'frame{idx}.jpg', base64.b64encode(img_encoded).decode('ascii'), 'image/jpeg'))
                    ]
                    # time = '0000_00_00_00_00_00_00'
                    time_ = now_time + dt.timedelta(seconds=0.033*idx)
                    time_ = time_.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-4]
                    url_full = self.url + f'?location={location}&time={time_}&frame={idx}'

                    while True:
                        try:
                            start = time.time()
                            print(url_full)
                            requests.request("POST", url_full, headers=headers, data=payload, files=files)

                            # print(response.text)
                            req_time = time.time() - start
                            print('request time : ', req_time)
                            idx += 1
                            avg_reference_time += req_time

                            break
                        except Exception as e:
                            print(e)
                            cv2.waitKey(1)

            url = f'http://{self.ip}:5000/releaseout'
            while True:
                try:
                    start = time.time()
                    print(url)
                    requests.request("GET", url, headers=headers, data=payload)

                    # print(response.text)
                    req_time = time.time() - start
                    print('request time : ', req_time)
                    break
                except Exception as e:
                    print(e)
                    cv2.waitKey(1)

            avg_reference_time /= idx
            res = make_response({'Status': 'Success', 'avg_request_time': avg_reference_time})
            res.headers['Content-type'] = 'application/json'
            return res

        res = make_response({'Status': 'Fail'})
        res.headers['Content-type'] = 'application/json'
        return res


api.add_resource(Index, '/', '/index')
# API를 간단히 설명해주는 페이지

api.add_resource(RequestVideo, '/reqvideo')
# Request video API

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='15000')
    # app.run()
