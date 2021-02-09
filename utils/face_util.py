import torch
import numpy as np
import cv2
import glob
import os
from utils.align_faces import warp_and_crop_face, get_reference_facial_points


class FaceFeatures(object):
    def __init__(self,root_path='./features'):
        paths = glob.glob(root_path+'/*.npy')
        self.feats = np.zeros((len(paths),1024),dtype=np.float32)
        self.names = []
        for i,path in enumerate(paths):
            self.feats[i] = load_feat(path)
            self.names.append(path.split('/')[-1].replace('.npy',''))

class Info(object):
    def __init__(self, box, sim, face_det_score,feature,patch,start_time):
        self.name = '???'
        self.cnt = 0
        self.sim = sim
        self.sim_score= 0
        self.box = box
        self.start_time = start_time
        self.end_time = start_time
        self.age = -1
        self.features = np.expand_dims(feature,0)
        self.patchs = patch
        self.face_det_score = face_det_score
    
    def update(self,Faces,sim_score,face_det_score,feature,patch,now_time,sim_threshold):
        self.features = np.concatenate([self.features,np.expand_dims(feature,0)],0)
        self.patch = np.concatenate([self.patchs,patch],1)
        self.sim += sim_score
        self.cnt+=1
        self.end_time = now_time
        self.face_det_score = face_det_score
        idx = np.argmax(self.sim / self.cnt)
        self.sim_score = self.sim[idx] / self.cnt        
        print("sim",sim_threshold)
        print(self.features.shape)
        if self.cnt == 5:
            if self.sim[idx] /5 >= sim_threshold:
                
                self.name = Faces.names[idx]                
            else:
                self.name = 'Unknown'

class TrackFace(object):
    def __init__(self,max_cnt=5,detect_threshold=0.8, sim_threshold=0.15,log_path='./log',day='0'):
        self.track_id = dict()
        self.max_cnt = max_cnt
        self.detect_threshold = detect_threshold
        self.sim_threshold = sim_threshold
        self.old_id = dict()
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_path = log_path
        self.day = day

    def update_day(self,day):
        self.day = day

    def draw_info(self,img_raw):
        for idx in self.track_id.keys():
            name = self.track_id[idx].name
            sim = self.track_id[idx].sim_score
            fs = self.track_id[idx].face_det_score
            box = self.track_id[idx].box
            cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(img_raw, f'name={name}', (box[0], box[1]+12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_raw, f'fs={fs:.2f}', (box[0], box[1]+24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_raw, f'sim={sim:.2f}', (box[0], box[1]+36),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            
            age = self.track_id[idx].age
            cv2.putText(img_raw, f'tracker age={age}', (box[0], box[1]+48),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    def write_info(self,info):
        file_name = os.path.join(self.log_path,self.day+'.csv')
        f = open(file_name,'w') if not os.path.exists(file_name) else open(file_name,'a')
        f.writelines(f'{info.name},{info.start_time},{info.end_time}\n')
    
    def age_update(self):
        for idx in list(self.old_id.keys()):
            self.old_id[idx].age +=1
            if self.old_id[idx].age >= 100:
                self.write_info(self.old_id[idx])         
                del self.old_id[idx]

        for idx in list(self.track_id.keys()):
            self.track_id[idx].age+=1
            if self.track_id[idx].age >= 71:
                self.old_id[idx] = self.track_id[idx]
                del self.track_id[idx]

    def check_identities(self,bbox,identities,now_time):
        ret_id, ret_box = [], []
        for i, id_num in enumerate(identities):
            id_num = int(id_num)
            if id_num in self.track_id.keys() and self.track_id[id_num].cnt >= self.max_cnt: 
                self.track_id[id_num].age = -1
                self.track_id[id_num].box = bbox[i]
                self.track_id[id_num].end_time = now_time
                print("checked")
                continue
            ret_id.append(id_num)
            ret_box.append(bbox[i])

        return ret_id,ret_box

    def update(self,Faces,identities, bbox_xyxy,face_boxes,face_det_scores,sim_scores,features,patches,now_time):
        ck_list = [0] * len(identities)
        for i in range(len(identities)):
            if identities[i] in self.track_id.keys():
                ck_list [i] = True
                id_num = identities[i]
                print("find")
                self.track_id[id_num].age = -1
                self.track_id[id_num].box = bbox_xyxy[i]
                if face_det_scores[i] > self.detect_threshold:
                    self.track_id[id_num].update(Faces,sim_scores[i],face_det_scores[i],features[i],patches[i],now_time,self.sim_threshold)
        
        for i in range(len(identities)):
            if ck_list[i] or face_det_scores[i] <= self.detect_threshold: continue
            best = 0
            best_id = -1
            sim = 0
            for idx in self.old_id.keys():
                sim = cosin_metric(features[i],self.old_id[idx].features,multi=True)
                sim = np.sum(sim) / self.old_id[idx].cnt
                if sim> best:
                    best = sim
                    best_id = idx
            if sim > self.sim_threshold:
                self.track_id[identities[i]] = self.old_id[best_id]
                del self.old_id[best_id]
                self.track_id[identities[i]].age = -1
                self.track_id[identities[i]].box = bbox_xyxy[i]
                if self.track_id[identities[i]].cnt < self.max_cnt :
                    self.track_id[identities[i]].update(Faces,sim_scores[i],face_det_scores[i],features[i],patches[i],now_time,self.sim_threshold)
            else:
                self.track_id[identities[i]] = Info(bbox_xyxy[i],  sim_scores[i],face_det_scores[i],features[i],patches[i],now_time)
                self.track_id[identities[i]].cnt+=1

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

def cosin_metric(x1, x2,multi=False):
    if multi:
        x2 = x2.transpose(1,0)
        n1 = np.linalg.norm(x1,axis=1,keepdims=True)
        n2 = np.linalg.norm(x2,axis=0,keepdims=True)
        return np.dot(x1,x2).squeeze() / (n1*n2)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def distinct_single_face(arc_model,target_feat,face_img):
    src_feat = get_face_feature(arc_model,face_img)
    sims = cosin_metric(src_feat,target_feat,multi=True)
    idx = np.argmax(sims,axis=1)
    return sims[0][idx[0]],idx[0]

def distinct_multi_face(arc_model,target_feat,patch):
    ps = None
    for i,p in enumerate(patch):
        p = cv2.resize(p,(128,128))
        p = cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
        p = np.dstack((p, np.fliplr(p)))
        p = p.transpose((2, 0, 1))
        p = p[:, np.newaxis, :, :]
        p = p.astype(np.float32, copy=False)
        p -= 127.5
        p /= 127.5
        if ps is None:
            ps = p
        else:
            ps = np.concatenate((ps, p), axis=0)

    feats = get_face_feature(arc_model,ps,preprocess=False)
    sims = cosin_metric(feats,target_feat,multi=True)
    idxs = np.argmax(sims,axis=1)
    return sims,idxs,feats

def get_face_feature(arc_model,image,preprocess=True):
    if preprocess:
        image = np.dstack((image, np.fliplr(image)))
        image = image.transpose((2, 0, 1))
        image = image[:, np.newaxis, :, :]
        image = image.astype(np.float32, copy=False)
        image -= 127.5
        image /= 127.5
    data = torch.from_numpy(image)
    data = data.to(torch.device("cuda"))

    output = arc_model(data)
    output = output.data.cpu().numpy()
    fe_1 = output[::2]
    fe_2 = output[1::2]

    return  np.hstack((fe_1, fe_2))


def save_feat(opt,name,feature,Faces):
    dst_path = f'{opt.features_path}/{name}.npy'
    feature = np.reshape(feature,[1,-1])
    Faces.feats = np.concatenate([Faces.feats,feature],0)
    Faces.names.append(name)
    np.save(dst_path,feature)

def load_feat(path):
    return np.load(path)

def make_patch_img(img_raw,bbox_xyxy):
    img_raw = np.float32(img_raw)
    img_batch = np.zeros((len(bbox_xyxy),img_raw.shape[0],img_raw.shape[1],img_raw.shape[2]),dtype=np.float32)
    for i,box in enumerate(bbox_xyxy):
        batch = np.zeros_like(img_raw,dtype = np.float32)
        batch[box[0]:box[2],box[1]:box[3]] = img_raw[box[0]:box[2],box[1]:box[3]]
        batch -= (104,117,123)
        img_batch[i] = batch
    img_batch = img_batch.transpose(0,3,1,2)
    return img_batch

def multi_batch_process(raw, detector, output_size, bbox_xyxy):
    img = make_patch_img(raw.copy(),bbox_xyxy)

    det, facial5points = detector.detect_multi_batch_faces(img)
    if not len(det):
        print("no face!")
    warp_and_crops = []
    for i in range(len(det)):
        if not len(facial5points[i]):
            warp_and_crops.append(np.zeros((128,128,3),np.float32))
            continue
        points = np.reshape(facial5points[i].copy(), (2, 5))
        
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)
        cropped_image = warp_and_crop_face(raw, points, reference_pts=reference_5pts, crop_size=output_size)
        warp_and_crops.append(cv2.resize(cropped_image,(128,128)))
    
    return det, np.array(warp_and_crops)

def process(raw, detector, output_size):
    img = raw.copy()
    det, facial5points = detector.detect_faces(img)
    if not len(det):
        print("no face!")
    warp_and_crops = []
    for i in range(len(det)):
        points = np.reshape(facial5points[i].copy(), (2, 5))
        
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)
        warp_and_crops.append(warp_and_crop_face(raw, points, reference_pts=reference_5pts, crop_size=output_size))
        
    return det, np.array(warp_and_crops)

def face_recognition_multi(img_raw,arc_model,face_detector,Faces,draw_img=False,indivisual_threshold=False, bbox_xyxy=None):
    boxes, det_scores, sim_scores, features,patches = [],[],[],[],[]
    img = img_raw.copy()
    if bbox_xyxy is None:
        det,patch = process(img,face_detector, output_size=(112, 112))
    else:
        det,patch = multi_batch_process(img,face_detector,output_size=(112,112),bbox_xyxy=bbox_xyxy)
        
    sims, idxs, feats = distinct_multi_face(arc_model,Faces.feats,patch)
    for i,b in enumerate(det):
        if len(b):
            detect_score = b[4]
            boxes.append(b[:4])
            det_scores.append(detect_score)
            sim_scores.append(sims[i])
            features.append(feats[i])
            patches.append(patch[i])

            if draw_img:
                sim , idx =  sims[i][idxs[i]], idxs[i]
                draw_name = Faces.names[idx]
                b = list(map(int, b))
                cx = b[0]
                cy = b[1] + 12
                cy2 = b[1] 
                cy3 = b[1] - 12
                cy4 = b[1] - 24
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                text = "detect={:.4f}".format(detect_score)
                text2 = "name={}".format(draw_name)
                cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.putText(img_raw, text2, (cx, cy2),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                text = "sim={:.4f} ".format(sim)
                text2 = "number={} ".format(i)
                
                cv2.putText(img_raw, text, (cx, cy3),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.putText(img_raw, text2, (cx, cy4),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        else:
            det_scores.append(0)
            sim_scores.append(np.zeros((Faces.feats.shape[0]),np.float32))
            features.append(np.zeros((1024),np.float32))
            patches.append(patch[i])
    if draw_img:
        return boxes, det_scores, sim_scores, features,patches,img_raw
    return boxes, det_scores, sim_scores,features,patches


def face_recognition(img_raw,arc_model,detector,Faces,detect_threshold=0.8,sim_threshold=0.125,draw_img=False,indivisual_threshold=False):
    names, boxes, det_scores, sim_scores = [],[],[],[]
    img = img_raw.copy()
    det,patch = process(img,detector, output_size=(112, 112))
    for i,b in enumerate(det):
        detect_score = b[4]
        b = list(map(int, b))

        p = cv2.resize(patch[i],(128,128))
        sim,idx = distinct_single_face(arc_model,Faces.feats,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
        draw_name = Faces.names[idx]
        if indivisual_threshold:
            if detect_score < detect_threshold:
                continue
            if sim < sim_threshold:
                draw_name = '???'
                sim = -1.
        else:
            if detect_score* sim < detect_threshold * sim_threshold:
                draw_name = '???'
                sim = -1.

        names.append(draw_name)
        boxes.append(b[:4])
        det_scores.append(detect_score)
        sim_scores.append(sim)

        if draw_img:
            b = list(map(int, b))
            cx = b[0]
            cy = b[1] + 12
            cy2 = b[1] 
            cy3 = b[1] - 12
            cy4 = b[1] - 24
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

            text = "detect={:.4f}".format(detect_score)
            text2 = "name={}".format(draw_name)
            cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_raw, text2, (cx, cy2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            text = "sim={:.4f} ".format(sim)
            text2 = "number={} ".format(i)
            
            cv2.putText(img_raw, text, (cx, cy3),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(img_raw, text2, (cx, cy4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    
    if draw_img:
        return names, boxes, det_scores, sim_scores, img_raw
    return names, boxes, det_scores, sim_scores