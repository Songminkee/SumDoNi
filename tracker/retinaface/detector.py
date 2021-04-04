from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn

#from retinaface.data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .loader import load_model
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms
import torchvision
import time

class RetinafaceDetector:
    def __init__(self, net='mnet', device_type='cuda',is_gpu=False,opt=None):
        cudnn.benchmark = True
        self.net = net
        self.device = torch.device(device_type)
        self.model = load_model(net,opt.cfg_mnet).to(self.device)
        self.model.eval()
        self.is_gpu = is_gpu
        self.priors = None
        self.prior_data =None
        self.cfg = opt.cfg_mnet
        self.confidence_threshold = opt.confidence_threshold
        self.nms_threshold = opt.nms_threshold

    def detect_faces(self, img_raw, top_k=5000,  keep_top_k=750, resize=1):
        start_time =time.time()
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
        if self.priors==None:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            self.priors = priors.to(self.device)
            self.prior_data = self.priors.data

        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        
        if self.is_gpu:
            # gpu
            scores = conf.squeeze(0)[:,1]
            landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize

            inds = torch.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            order = scores.argsort()
            order = torch.from_numpy(order.cpu().numpy()[::-1].copy())[:top_k]
            
            boxes = boxes[order]
            scores = scores[order]
            landms = landms[order]
            keep = torchvision.ops.boxes.nms(boxes, scores, self.nms_threshold).cpu().numpy()
            
            dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis].cpu().numpy())).astype(np.float32, copy=False)
            dets = dets[keep]
            landms = landms.cpu().numpy()[keep]
        else:
            # cpu
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            

            landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            
            # keep top-K before NMS
            order = scores.argsort()[::-1][:top_k]
            
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]


            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            
            keep = py_cpu_nms(dets, self.nms_threshold)
           
            
            dets = dets[keep, :]
            landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        
        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )
        return dets, landms

    def detect_multi_batch_faces(self, img_raw, top_k=1,  keep_top_k=1, resize=1):
        img = np.float32(img_raw)
        im_height, im_width = img[0].shape[-2:]
        scale = torch.Tensor([im_width, im_height,im_width, im_height])
        img = torch.from_numpy(img).to(self.device)
        scale = scale.to(self.device)
        
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
        

        if self.priors==None:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            self.priors = priors.to(self.device)
            self.prior_data = self.priors.data

        boxes = [decode(loc[i].data.squeeze(0), self.prior_data, self.cfg['variance']) for i in range(len(loc))]
        boxes = [boxes[i] * scale / resize for i in range(len(boxes))]
        
        ret_det, ret_facial5points = [], []

        for b,c,l in zip(boxes,conf,landms):
            # cpu
            b = b.cpu().numpy()
            s = c.squeeze(0).data.cpu().numpy()[:, 1]
            

            l = decode_landm(l.data.squeeze(0), self.prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            l = l * scale1 / resize
            l = l.cpu().numpy()

            # ignore low scores
            inds = np.where(s > self.confidence_threshold)[0]
            b = b[inds]
            l = l[inds]
            s = s[inds]
            
            # keep top-K before NMS
            order = s.argsort()[::-1][:top_k]
            
            b = b[order]
            l = l[order]
            s = s[order]


            # do NMS
            dets = np.hstack((b, s[:, np.newaxis])).astype(np.float32, copy=False)
            
            keep = py_cpu_nms(dets, self.nms_threshold)
            
            
            dets = dets[keep, :]
            l = l[keep]

            # keep top-K faster NMS
            dets = dets[:keep_top_k, :]
            l = l[:keep_top_k, :]
            
            l = l.reshape((-1, 5, 2))
            l = l.transpose((0, 2, 1))
            l = l.reshape(-1, 10, )
            ret_det.append(dets.reshape(-1))
            ret_facial5points.append(l)
        return ret_det, ret_facial5points