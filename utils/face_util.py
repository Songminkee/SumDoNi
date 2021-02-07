import torch
import numpy as np
import cv2
import glob
from utils.align_faces import warp_and_crop_face, get_reference_facial_points


class FaceFeatures(object):
    def __init__(self,root_path='./features'):
        paths = glob.glob(root_path+'/*.npy')
        self.feats = np.zeros((len(paths),1024),dtype=np.float32)
        self.names = []
        for i,path in enumerate(paths):
            self.feats[i] = load_feat(path)
            self.names.append(path.split('/')[-1].replace('.npy',''))

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
        x2 = np.reshape(x2,(-1,1))
        n1 = np.linalg.norm(x1,axis=1)
        n2 = np.linalg.norm(x2,axis=0)
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

def distinct_face(arc_model,target_feat,face_img):
    src_feat = get_face_feature(arc_model,face_img)
    sims = cosin_metric(target_feat,src_feat.squeeze())
    idx = np.argmax(sims)
    return sims[idx],idx

def get_face_feature(arc_model,image):
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

def face_recognition(img_raw,arc_model,detector,Faces,detect_threshold=0.8,sim_threshold=0.125,draw_img=False,indivisual_threshold=False):
    names, boxes, det_scores, sim_scores = [],[],[],[]
    img = img_raw.copy()
    det,patch = process(img,detector, output_size=(112, 112))
    for i,b in enumerate(det):
        detect_score = b[4]
        b = list(map(int, b))

        p = cv2.resize(patch[i],(128,128))
        sim,idx = distinct_face(arc_model,Faces.feats,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
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