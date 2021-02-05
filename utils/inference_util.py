import torch
import numpy as np

from utils.align_faces import warp_and_crop_face, get_reference_facial_points

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