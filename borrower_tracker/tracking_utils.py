import os
import sys
sys.path.insert(0, './tracker/yolov5')
# https://github.com/pytorch/pytorch/issues/3678#issuecomment-474076294

from tracker.config import Config
from tracker.arcface.models import resnet_face18
from tracker.utils.general import non_max_suppression, scale_coords
from tracker.utils.face_util import load_model, FaceFeatures, TrackFace, \
                                    make_feature, face_recognition_multi
from tracker.utils.detect_util import bbox_rel
from tracker.retinaface.detector import RetinafaceDetector
from tracker.deep_sort import DeepSort
from asgiref.sync import sync_to_async
from django.utils import timezone

from datetime import datetime
from collections import deque
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import asyncio
import gc
import skvideo.io


class TrackingModels:

    def __init__(self):
        self.config = Config()
        self.arc_model, self.face_detector, self.deepsort, \
        self.person_detector, self.Faces = \
            self.load_models()

    def load_models(self):
        '''Load tracking models.

        Args:
            
        Returns:
            arc_model (obj): arcface model
            face_detector (obj): face detection model
            deepsort (obj): face tracking model
            person_detector (obj): person detection model
            Faces (obj)
        '''
        # Set configurations
        opt = Config()
        cudnn.benchmark = True
        device = torch.device("cuda")

        # load arcface model
        arc_model = resnet_face18(opt.use_se)
        load_model(arc_model, opt.arc_model_path)
        arc_model.to(device)
        arc_model.eval()

        # load retina face model (face detector)
        face_detector = RetinafaceDetector(is_gpu=False, opt=opt)    

        # load deep sort model
        deepsort = DeepSort(opt.deepsort_cfg.DEEPSORT.REID_CKPT,
                            max_dist=opt.deepsort_cfg.DEEPSORT.MAX_DIST, min_confidence=opt.deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=opt.deepsort_cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=opt.deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=opt.deepsort_cfg.DEEPSORT.MAX_AGE, n_init=opt.deepsort_cfg.DEEPSORT.N_INIT, nn_budget=opt.deepsort_cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # load yolo5 model (person detector)
        person_detector = torch.load(opt.yolo_model_path, map_location=device)['model'].float().half()

        # load face featrues
        Faces = FaceFeatures(opt.features_path)

        return arc_model, face_detector, deepsort, person_detector, Faces

    async def find_tlog(self, t_logs, t_info_b_name):
        '''Find and return tracked log index.'''
        for i, (_, borrower, t_log) in enumerate(t_logs):
            if borrower.b_name == t_info_b_name:
                return t_log
        return None

    async def save_tracked_videos(self, tracked_ids, user, Borrower,
                                  UserBorrower, TrackingLog, BorrowerTrackingLog,
                                  resize, write_size, vqueue, vq_tidxes):
        '''Save tracking information into Database.
        
        Args:
            tracked_ids (dict): tracked ids
            Borrower (obj): Borrower model
            UserBorrower (obj): UserBorrower model
            TrackingLog (obj): TrackingLog model
            BorrowerTrackingLog (obj): BorrowerTrackingLog model
            resize (int): size of a resize for an image
            write_size (int): writed image size
            vqueue (deque): video queue for tracked information
            vq_tidxes (dict): start vqueue index for each tracked id

        Returns:
            no returns
        '''
        t_logs = []
        for tracked_id, t_info in tracked_ids.items():
            # Remove Unknown and ???
            if t_info.name == 'Unknown' or t_info.name == '???':
                if vq_tidxes.get(tracked_id):
                    del vq_tidxes[tracked_id]
                continue

            # Find a borrower
            # https://docs.djangoproject.com/en/3.2/topics/async/#async-safety
            user_borrowers = await sync_to_async(UserBorrower.objects.filter,
                                                 thread_sensitive=True)(uid=user)
            for user_borrower in user_borrowers:
                borrower = await sync_to_async(Borrower.objects.get,
                                               thread_sensitive=True)(bid=user_borrower.bid.bid,
                                                                      b_name=t_info.name)
                if borrower:
                    break
            
            # Merge duplicates
            prev_tlog = await self.find_tlog(t_logs, t_info.name)
            if prev_tlog:
                t_log = prev_tlog
                if t_log.start_datetime <= t_info.start_time <= t_log.end_datetime \
                and t_log.end_datetime < t_info.end_time:
                    t_log.end_datetime = t_info.end_time
                elif t_log.start_datetime <= t_info.end_time <= t_log.end_datetime \
                and t_info.start_time < t_log.start_datetime:
                    t_log.start_datetime = t_info.start_time

                # Update tracked log information
                del vq_tidxes[tracked_id]
                await sync_to_async(t_log.save, thread_sensitive=True)()
            # Create new tracked log
            else:
                t_log = await sync_to_async(TrackingLog.objects.create,
                                            thread_sensitive=True)(start_datetime=t_info.start_time,
                                                                end_datetime=t_info.end_time,
                                                                video_path='')
                await sync_to_async(t_log.save, thread_sensitive=True)()
                t_logs.append((tracked_id, borrower, t_log))

        # Sort tracked logs
        t_logs.sort(key=lambda x: vq_tidxes[x[0]])

        # Save tracking information
        for i, (tracked_id, borrower, t_log) in enumerate(t_logs):
            bt_log = await sync_to_async(BorrowerTrackingLog.objects.create,
                                         thread_sensitive=True)(bid=borrower,
                                                                tid=t_log)
            await sync_to_async(bt_log.save, thread_sensitive=True)()

            # Create output directory path
            out_dir = f'static/log/{user.uid}/{borrower.bid}'
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            # Save a video path
            file_path = f'{out_dir}/{t_log.start_datetime}-{t_log.end_datetime}.mp4'
            t_log.video_path = file_path.replace('static/', '')
            await sync_to_async(t_log.save, thread_sensitive=True)()

            # Create a VideoWriter to save Video
            fps = 30
            fcc = cv2.VideoWriter_fourcc(*'FMP4')
            out = skvideo.io.FFmpegWriter(file_path, outputdict={'-vcodec': 'libx264'})

            # Save frames
            for frame in list(vqueue):
                frame = cv2.cvtColor(cv2.resize(frame, (write_size, write_size),
                                                interpolation=cv2.INTER_LINEAR),
                                     cv2.COLOR_BGR2RGB)
                out.writeFrame(frame)

            # Release VideoWriter
            out.close()
            out = None

            # Update vqueue
            if len(t_logs) != (i+1):
                next_tracked_id = t_logs[i+1][0]
                next_start_idx = vq_tidxes[next_tracked_id]
                print('i, next_start_idx:', i, next_start_idx)
                for j in range(next_start_idx):
                    vqueue.popleft()

                for j in range(i+1, len(t_logs)):
                    tracked_id = t_logs[j][0]
                    vq_tidxes[tracked_id] -= next_start_idx

    async def track_face(self, cap, User=None, username='', is_write=True,
                         result_name='test', write_size=416, is_resize=True,
                         resize=416, indivisual_threshold=False, Borrower=None,
                         UserBorrower=None, TrackingLog=None, BorrowerTrackingLog=None):
        ''' Track faces.
        
            Args:
                (WIP)

            Returns:
                no returns.
        '''
        tracked = TrackFace(max_cnt=5,
                            detect_threshold=self.config.detect_threshold,
                            sim_threshold=self.config.sim_threshold)

        frame = 0
        vqueue, vq_tidxes, VQ_MAX = deque([]), {}, 90
        while True:
            # Check a tracking status!!
            # If the tracking status is False, then makes tracking off!!
            user = await sync_to_async(User.objects.get, thread_sensitive=True)(username=username)
            print('User Status:', user.tracking_status)
            if user.tracking_status is False:
                print('Removed all tracked ids using Tracking Off!!')
                cap.release()

                # Save tracked info for tracked ids
                await self.save_tracked_videos(tracked.track_id, user, Borrower,
                                               UserBorrower, TrackingLog, BorrowerTrackingLog,
                                               resize, write_size, vqueue, vq_tidxes)
                
                # Save tracked info for old ids
                await self.save_tracked_videos(tracked.old_id, user, Borrower,
                                               UserBorrower, TrackingLog, BorrowerTrackingLog,
                                               resize, write_size, vqueue, vq_tidxes)

            # Extract a frame from CCTV
            t1 = time.time()
            ret, img_raw = cap.read()
            if not ret:
                break

            # Resize an image
            img_raw = cv2.resize(img_raw, (resize, resize),
                                 interpolation=cv2.INTER_LINEAR)
            
            # Preprocess the image
            img = torch.from_numpy(np.float32(img_raw.copy()).transpose(2,0,1))
            img = img.to('cuda').half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Detect a person
            t2 = time.time()
            pred = self.person_detector(img, augment=False)[0]
            det = non_max_suppression(pred, self.config.yolo_confidence_threshold,
                                      self.config.yolo_nms_threshold,
                                      classes=[0], agnostic=False)[0]
            print(f"detection,nms time = {time.time()-t2}")

            if len(det) : 
                t3 = time.time()

                # Extract bounding boxes and confidence scores
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
                print(f"deepsort preprocessing time = {time.time()-t3}")

                # Enroll detections into deepsort
                t4 = time.time()
                outputs = self.deepsort.update(xywhs, confss, img_raw)
                print(f"deepsort time = {time.time()-t4}")

                if len(outputs):
                    # Update only an end time if bboxes is already exited
                    # But if not, extract the bboxes
                    t5 = time.time()
                    identities, bbox_xyxy = tracked.check_identities(outputs[:,:4],
                                                                     outputs[:,-1],
                                                                     str(timezone.localtime()))
                    print(f"check_id time = {time.time()-t5}")

                    # Recognize faces and update tracking information
                    if len(bbox_xyxy):
                        # Extract borrower features of this user
                        face_features = []
                        for uid, feature in zip(self.Faces.uids, self.Faces.feats):
                            if user.uid == uid:
                                face_features.append(feature)
                        face_features = np.asarray(face_features)

                        # Recognize Faces
                        t6 = time.time()
                        face_boxes, face_det_scores, sim_scores, features, patches, idxs = \
                            face_recognition_multi(img_raw, self.arc_model,
                                                   self.face_detector, face_features,
                                                   indivisual_threshold=indivisual_threshold,
                                                   bbox_xyxy=bbox_xyxy)
                        print(f"face detect,recog time = {time.time()-t6}")

                        # Update tracking information
                        t7 = time.time()                  
                        tracked.update(self.Faces, identities, bbox_xyxy,
                                       face_boxes, face_det_scores, sim_scores,
                                       features, patches, str(timezone.localtime()))
                        print(f"update time = {time.time()-t7}")

                        print('Faces.names', self.Faces.names)
                        print('tracked.track_id:', tracked.track_id)
                        print('tracked_names:', [self.Faces.names[idx] for idx in idxs])

                        # Save start idx in vqueue for the tracked idx
                        print('identities:', identities)
                        for idx in identities:
                            if tracked.track_id and tracked.track_id.get(idx) and tracked.track_id[idx].cnt == 5:
                                vq_tidxes[idx] = max(0, len(vqueue) - VQ_MAX)

            # Update a tracked age
            frame += 1
            t9 = time.time()
            removed_old_ids = tracked.age_update()
            print(f"age time = {time.time()-t9}")
            if removed_old_ids:
                print('Removed old tracked ids!!')
                self.save_tracked_videos(removed_old_ids, user, Borrower,
                                         UserBorrower, TrackingLog, BorrowerTrackingLog,
                                         resize, write_size, vqueue, vq_tidxes)
                del removed_old_ids
                gc.collect()

            # Write a video
            if is_write:
                img_draw = img_raw.copy()

                t8 = time.time()
                tracked.draw_info(img_draw)
                print(f"draw time = {time.time()-t8}")
                
                cv2.putText(img_draw, f'now_frame={frame}', (0, 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

                # Append a frame into vqueue
                if len(vqueue) == VQ_MAX and not tracked.is_tracked():
                    vqueue.popleft()
                vqueue.append(cv2.resize(img_draw, (resize, resize)))
            
            print('tracked',tracked)
            print(f"time = {time.time()-t1}\n")
            await asyncio.sleep(0.0001)
        
        # Release a connection
        cap.release()

        for t in tracked.track_id.values():
            print('Borrower name:', t.name)
            print(f'Tracked time: {t.start_time} ~ {t.end_time}')

        for t in tracked.old_id.values():
            print('Borrower name:', t.name)
            print(f'Tracked time: {t.start_time} ~ {t.end_time}')

        # This is for video cctv.
        await self.save_tracked_videos(tracked.old_id, user, Borrower,
                                       UserBorrower, TrackingLog, BorrowerTrackingLog,
                                       resize, write_size, vqueue, vq_tidxes)
