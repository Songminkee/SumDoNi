import torch
import numpy as np
import cv2
import glob
import os
from utils.align_faces import warp_and_crop_face, get_reference_facial_points


class FaceFeatures(object):
    """ This is a class that has saved facial features.

    Params:
        feats ([numpy array]) : It has features for N images. N x 1024.
        name ([string list]) : It is a name to distinguish N features.

    __init__:
        Args:
            root_path ([string]) : This is the root path of the features you have saved.
    """
    def __init__(self,root_path='./features'):
        folders = glob.glob(root_path+'/*')
        paths = []
        for folder in folders:
            files = glob.glob(folder+'/*.npy')
            for file in files:
                paths.append(file.replace("\\","/"))
        self.feats = np.zeros((len(paths),1024),dtype=np.float32)
        self.names = []
        for i,path in enumerate(paths):
            loaded =load_feat(path)
            self.feats[i] = loaded
            self.names.append(path.split('/')[-2])

class Info(object):
    """ It is a class that has information about similarity score, name, feature, box, etc. for one object.
    
    Params:
        box ([int list]) : Box information for person object detection. xyxy format.
        sim ([numpy array]) : This object's similarity score for all features of the FaceFeatures class. Shape is N.
        sim_score ([float]) : The highest value of sim/cnt.
        cnt ([int]) : Counted number of times to perform face detection to calculate the final similarity.
        start_time ([string]) : The first detection time.
        end_time ([string]) : The last detection time.
        age ([int]) : Undetected frames from the last detected frame. 
                        When this number exceeds a certain range, the memory of the class is released.
        features ([numpy array]) : The face feature of this object. Max shape is max_cnt x 1024.
        face_det_score ([float]) : Final confidence score for face detection.

    Methods:
        __init__:
            Args:
                box ([int list]) : Box information for person object detection. xyxy format.
                sim ([numpy array]) : This object's similarity score for all features of the FaceFeatures class. Shape is N.
                face_det_score ([float]) : Final confidence score for face detection.
                feature ([numpy array]) : The face feature of this object.
                patch ([numpy array]) : The patch image of detected face.
                sim_score ([float]) : The highest value of sim/cnt.
                start_time ([string]) : The first detection time.

        update:  This function is called when this object is detected at the current time, 
                    and is named by similarity if it is greater than or equal to max_cnt specified by cnt.
            Args:
                Faces : FaceFeatures class object.
                sim_score ([numpy array]) : This object's similarity score for all features of the FaceFeatures class. Shape is N.
                face_det_score ([float]) : Final confidence score for face detection. 
                feature ([numpy array]) : The face feature of this object.
                patch ([numpy array]) : The patch image of detected face.
                now_time ([float]) : The current time is entered to update the end time.
                sim_threshold ([float]): Threshold value for maximum similarity score.

    """
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
    
    def __repr__(self):
        information = f"""        name : {self.name}
        cnt : {self.cnt}
        sim : {self.sim}
        sim_score : {self.sim_score}
        box : {self.box}
        time : {[self.start_time,self.end_time]}
        age : {self.age}
        det_score : {self.face_det_score}\n"""
        return information

    def update(self,Faces,sim_score,face_det_score,feature,patch,now_time,sim_threshold):
        self.features = np.concatenate([self.features,np.expand_dims(feature,0)],0)
        self.patch = np.concatenate([self.patchs,patch],1)
        self.sim += sim_score
        self.cnt+=1
        self.end_time = str(now_time)
        self.face_det_score = face_det_score
        idx = np.argmax(self.sim / self.cnt)
        self.sim_score = self.sim[idx] / self.cnt        
        if self.cnt == 5:
            if self.sim[idx] /5 >= sim_threshold:
                
                self.name = Faces.names[idx]                
            else:
                self.name = 'Unknown'
    


class TrackFace(object):
    """ It is a class that has information about similarity score, name, feature, box, etc. for one object.
    
    Params:
        track_id ([Dictionary (key = integer, value = Info class) ]) : It is a dictionary that has an ID currently being tracked as a key and an Info class with a value.
        max_cnt ([int]) : Total number of times to perform face detection to calculate the final similarity.
        detect_threshold ([float]) : Threshold value of face detection.
        sim_threshold ([float]) : Threshold value for maximum similarity score.
        old_id ([Dictionary (key = integer, value = Info class) ]) : The info class whose age has risen to a certain level is stored in this dictionary, 
                                                                                            where the memory is released when age exceeds a certain number.
        age ([int]) : Undetected frames from the last detected frame. 
                        When this number exceeds a certain range, the memory of the class is released.
        log_path ([string]): The root path of the log to be left when memory is released.
        day ([numpy array]) : This variable is still temporary. This is a variable set to record the log by date.
       
    Methods:
        draw_info: This function draws information about the ID currently being tracked on the image.
            Args:
                img_raw ([numpy array]) : Original image.
        
        write_info: It is called when the memory of the info class is released, and the start time and end time are logged.
            Args:
                Info ([Info class]) : Info class to release memory.

        age_update: It is called every frame and increases the age by 1 in all Info classes.
            Args:
                None, just call.

        check_identities: Called before performing face recognition. 
                                IDs that have already been recognized by max_cnt or higher and have been named are excluded.
                                These ids only update the end time, person detection box, and age.
            Args:
                bbox ([numpy array]) : Bounding box of Person detection. xyxy format.
                identities ([numpy array]) : Tracking ID received from Deep Sort.
                now_time ([numpy array]) : Current frame or time.

        update:  This function is called when this object is detected at the current time.
                    If the ID is already tracking, the update method of the Info class is called.
                    In the case of a new id, it may be the case that the old id is missed and a new id is given,
                    so if the feature is similar to the old id and exceeds the threshold, 
                    the old id is moved back to the track id.
                    If it is less than the threshold, it is determined as a new object and an Info class is created.
            Args:
                Faces : FaceFeatures class object.
                identities ([numpy array]) : Tracking ID received from Deep Sort.
                bbox_xyxy ([numpy array]) : Bounding box of Person detection. xyxy format.
                face_boxes ([numpy array]) : Bounding box of face detection. xyxy format.
                face_det_score ([float]) : Final confidence score for face detection. 
                sim_score ([numpy array]) : This object's similarity score for all features of the FaceFeatures class. Shape is N.
                feature ([numpy array]) : The face feature of this object.
                patch ([numpy array]) : The patch image of detected face.
                now_time ([float]) : The current time is entered to update the end time.
    """
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
    
    def __repr__(self):
        information = ''
        for k in self.track_id.keys():
            information += f'\ntracked_id : {k}\n' + self.track_id[k].__repr__()
        for k in self.old_id.keys():
            information += f'\nold_id : {k}\n' + self.old_id[k].__repr__()
        return information

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
            if self.old_id[idx].age >= 500:
                self.write_info(self.old_id[idx])         
                del self.old_id[idx]

        for idx in list(self.track_id.keys()):
            self.track_id[idx].age += 1
            if self.track_id[idx].age >= 10:
                self.old_id[idx] = self.track_id[idx]
                del self.track_id[idx]

    def check_identities(self,bbox,identities,now_time):
        ret_id, ret_box = [], []
        for i, id_num in enumerate(identities):
            id_num = int(id_num)
            if id_num in self.track_id.keys() and self.track_id[id_num].cnt >= self.max_cnt: 
                self.track_id[id_num].age = -1
                self.track_id[id_num].box = bbox[i]
                self.track_id[id_num].end_time = str(now_time)
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
            print("face_score",face_det_scores[i])
            if ck_list[i] or face_det_scores[i] <= self.detect_threshold: continue
            best = 0
            best_id = -1
            sim = 0
            for idx in self.old_id.keys():
                sim = cosin_metric(features[i][np.newaxis,:],self.old_id[idx].features,multi=True)
                sim = np.sum(sim) / self.old_id[idx].cnt
                if sim> best:
                    best = sim
                    best_id = idx
            print("sim",sim)
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
    """This is a function for the arc face model. Load and apply state dict from the saved file.

    Args:
        model ([ResNetFace]): Arc Face model class object.
        model_path ([string]): Path of ArcFace's pretrained weight file.
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def cosin_metric(x1, x2,multi=False):
    """Measure the similarity between two features using the cosin function.

    Args:
        x1 ([numpy array]): Shape is N x 1024 if multi. else 1024.
        x2 ([numpy array]): Shape is M x 1024 if multi. else 1024.
        multi (bool, optional): Flag for shape.

    Returns:
        [numpy]: Similarity for N features and M features if multi. else single.
    """
    if multi:
        x2 = x2.transpose(1,0)
        n1 = np.linalg.norm(x1,axis=1,keepdims=True)
        n2 = np.linalg.norm(x2,axis=0,keepdims=True)
        return np.dot(x1,x2).squeeze() / (n1*n2)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def distinct_single_face(arc_model,target_feat,face_img):
    """It extracts a feature from one image and returns which feature is most similar to one of the features you currently have.

    Args:
        arc_model ([ResNetFace class object]): Arc Face model class object.
        target_feat ([numpy array]): Features that you currently have.
        face_img ([numpy array]): Face image from which to extract features.
    Returns:
        [numpy array], [numpy array] : Highest similarity and corresponding id
    """
    src_feat = get_face_feature(arc_model,face_img)
    sims = cosin_metric(src_feat,target_feat,multi=True)
    idx = np.argmax(sims,axis=1)
    return sims[0][idx[0]],idx[0]

def distinct_multi_face(arc_model,target_feat,patch):
    """It extracts features from multiple patch images and returns the similarity with the current feature, the corresponding id, and the extracted feature.

    Args:
        arc_model ([ResNetFace class object]): Arc Face model class object.
        target_feat ([numpy array]): Features that you currently have.
        face_img ([numpy array]): Face image patches from which to extract features.
    Returns:
        [numpy array, numpy array, numpy array] : Highest similarity, corresponding id and extracted features
    """
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
    """Returns the arcface result from the input image.

    Args:
        arc_model ([ResNetFace class object]): Arc Face model class object.
        image ([numpy array]): Face image from which to extract features.
        preprocess (bool, optional): If you have already preprocessed the image, just enter False. Defaults to True.

    Returns:
        [numpy array] : Feature for face image.
    """
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


def save_feat(opt,name,feature,Faces,patch=None):
    """Save the face feature in npy format.

    Args:
        opt ([Config class]): Config class.
        name ([string]): The name of the face feature.
        feature ([numpy array]): Face feature.
        Faces ([FaceFeatures class]): FaceFeatures class object. Takes as input to update the current state.
        patch ([numpy array], optional): When input, the image patch from which the feature is extracted is also saved.
                                                         The name is the same as npy. Defaults to None.
    """
    if not os.path.exists(os.path.join(opt.features_path,name)):
        os.makedirs(os.path.join(opt.features_path,name))
    exists = glob.glob(os.path.join(opt.features_path,name)+'/*.npy')
    dst_path = f'{opt.features_path}/{name}/{name}_{len(exists)+1}.npy'
    feature = np.reshape(feature,[1,-1])
    Faces.feats = np.concatenate([Faces.feats,feature],0)
    Faces.names.append(name)
    np.save(dst_path,feature)
    if patch is not None:
        cv2.imwrite(f'{opt.features_path}/{name}/{name}_{len(exists)+1}.jpg',patch)

def load_feat(path):
    return np.load(path)

def make_patch_img(img_raw,bbox_xyxy):
    """Create an image set of areas detected as humans by person detection in the original image.

    Args:
        img_raw ([numpy array]) : Original image. h x w  x c.
        bbox_xyxy ([numpy array]) : Bounding box of Person detection. xyxy format.
                
    Returns:
       ([numpy array]): A set of patch images in batch format. b x c x h x w. 
    """
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
    """ A patch is created from the input image and the person detection box, 
        and face detection and face alignment are performed for each patch.

    Args:
        raw ([numpy array]) : Original image.
        detector ([RetinafaceDetector class]): Face Detector.
        output_size ([(int,int)]): Output size of face patches.
        bbox_xyxy ([numpy array]): Person detection boxes to make input patches

    Returns:
        [numpy array, numpy array]: Box information and result image of face detection.
    """
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
    """Perform face detection (Refinaface) on a single image and return results for it.

    Args:
        raw ([numpy array]) : Original image.
        detector ([RetinafaceDetector class]): Face Detector.
        output_size ([(int,int)]): Output size of face patches.

    Returns:
        [numpy array, numpy array]: Box information and result image of face detection.
    """
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
    """The difference with the face_recognition function is that it creates multiple patch images based on
        person dection information before face detection is performed,
        and face detection and recognition are calculated in batch form at once. 
        To do this, face detection acquires only one box with the highest confidence in each image.

    Args:
        img_raw ([numpy array]) : Original image.
        arc_model ([ResNetFace class object]): Arc Face model class object.
        face_detector ([RetinafaceDetector class]): Face Detector.
        Faces ([FaceFeatures class]): FaceFeatures class object. Takes as input to update the current state.
        draw_img (bool, optional): In the case of ture, the result is drawn with the image. Defaults to False.
        indivisual_threshold (bool, optional): Currently not used. It may be deleted in the future or the function may be changed. Defaults to False.
        bbox_xyxy ([numpy array], optional): Bounding box of Person detection. xyxy format. Defaults to None.

    Returns:
        [numpy array]: Box information of face detection,
                                Confidence score of face detection,
                                Similarity for all features you currently have,
                                Feature array about patch(face image).
                                Face patch images from which features were extracted.
    """
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
    """Face recognition is performed after face detection on the image.
        And it is compared with the index with the highest number compared to the current features and given a name. 
        Of course, above the threshold, the name is set to ???.

    Args:
        img_raw ([numpy array]) : Original image.
        arc_model ([ResNetFace class object]): Arc Face model class object.
        detector ([RetinafaceDetector class]): Face Detector.
        Faces ([FaceFeatures class]): FaceFeatures class object. Takes as input to update the current state.
        detect_threshold (float, optional): Threshold value of face detection. Defaults to 0.8.
        sim_threshold (float, optional): Threshold value for maximum similarity score. Defaults to 0.125.
        draw_img (bool, optional): In the case of ture, the result is drawn with the image.. Defaults to False.
        indivisual_threshold (bool, optional): This is the policy for the threshold.. Defaults to False.

    Returns:
        [string, numpy array, np.float32, np.float32]: name, face detection box information, face detection confidence score, maximum similarity
    """
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

def make_feature(opt,Faces,arc_model,face_detector,img_raw):
    """This function is called when you want to extract and save a feature from the original image.

    Args:
        opt ([Config class]): Config class.
        Faces ([FaceFeatures class]): FaceFeatures class object. Takes as input to update the current state.
        arc_model ([ResNetFace class object]): Arc Face model class object.
        face_detector ([RetinafaceDetector class]): Face Detector.
        img_raw ([numpy array]) : Original image.
    """
    is_ok = False
    det,patch = process(img_raw,face_detector, output_size=(112, 112))
    
    if not len(det): return
    candidate_img = img_raw.copy()
    for i,b in enumerate(det):
        b = list(map(int, b))
        cv2.rectangle(candidate_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        text = "number={} ".format(i)
        cv2.putText(candidate_img, text, (b[0], b[1]+12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255,0))
    
    while not is_ok:
        cv2.imshow("candidate",candidate_img)
        cv2.waitKey(1)
        number = int(input("input feature number (if you want break insert -1) : "))
        if number==-1:
            break
        patch_img = cv2.resize(patch[number],(128,128))
        draw_patch = patch_img.copy()
        cv2.putText(draw_patch, "is it right? (Y/N) ", (0, 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv2.imshow("your choice",draw_patch)
        k = cv2.waitKey(1)
        answer = input("is it right? (Y/N) : ")
        if answer=='Y' or answer=='y':
            name = input("input feature name : ")
            feat = get_face_feature(arc_model,cv2.cvtColor(patch_img,cv2.COLOR_BGR2GRAY),preprocess=True)
            save_feat(opt,name,feat,Faces,patch_img)
            
            answer = input("Will you keep saving it? (Y/N) ")
            if answer=='N':
                is_ok=True
    cv2.destroyWindow("candidate")
    cv2.destroyWindow('your choice')

