# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
import matplotlib.pyplot as plt
import glob

from config import Config
from arcface.models import resnet_face18
from retinaface.detector import RetinafaceDetector
from utils.inference_util import *

class FaceFeatures(object):
    def __init__(self,root_path='./features'):
        paths = glob.glob(root_path+'/*.npy')
        self.feats = np.zeros((len(paths),1024),dtype=np.float32)
        self.names = []
        for i,path in enumerate(paths):
            self.feats[i] = load_feat(path)
            self.names.append(path.split('/')[-1].replace('.npy',''))

def img_mode(config,args,arc_model,detector,Faces):
    img_raw = cv2.imread(args.src_path)
    if args.is_resize:
        img_raw = cv2.resize(img_raw, (args.resize, args.resize),interpolation=cv2.INTER_LINEAR)
    img = img_raw.copy()
    det,patch = process(img,detector, output_size=(112, 112))
    ps = []
    
    for i,p in enumerate(patch):
        p = cv2.resize(p,(128,128))
        feat = get_face_feature(arc_model,p)
        ps.append(p)
    
    for i,b in enumerate(det):
        detect_score = b[4]
        b = list(map(int, b))
        cx = b[0]
        cy = b[1] + 12
        cy2 = b[1] 
        cy3 = b[1] - 12
        cy4 = b[1] - 24
        
        p = cv2.resize(patch[i],(128,128))
        sim,idx = distinct_face(arc_model,Faces.feats,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
        
        draw_name = Faces.names[idx]
        if args.indivisual_threshold:
            if detect_score < opt.detect_threshold:
                continue
            if sim < opt.sim_threshold:
                draw_name = '???'
                sim = -1.
        else:
            if detect_score* sim < opt.detect_threshold * opt.sim_threshold:#sim < 0.15 :
                draw_name = '???'
                sim = -1.
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
    plt.show()

    cv2.putText(img_raw, 'Save feature of face, Press "s"', (0, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    cv2.imshow("raw",img_raw)
    if args.write:
        cv2.imwrite(f'{args.result_name}.jpg',img_raw)
        
    k = cv2.waitKey(0)
    if k==115:
        is_ok = False
        while not is_ok:
            number = int(input("input feature number : "))
            p = cv2.cvtColor(cv2.resize(patch[number],(128,128)),cv2.COLOR_BGR2RGB)
            plt.imshow(p)
            plt.show()
            answer = input("is it right? (Y/N) : ")
            if answer=='Y':
                name = input("input feature name : ")
                feat = get_face_feature(arc_model,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
                save_feat(opt,name,feat,Faces)
                answer = input("Will you keep saving it? (Y/N)")
                if answer=='N':
                    is_ok=True
    cv2.destroyAllWindows()

def video_mode(config,args,arc_model,detector,Faces):
    cap = cv2.VideoCapture(args.src_path)

    if args.write:
        fps=30
        fcc = cv2.VideoWriter_fourcc(*'FMP4')#cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') 
        out = cv2.VideoWriter(f'{args.result_name}.mp4', fcc, fps, (args.write_size,args.write_size))
    
    while True:
        ret, img_raw = cap.read()
        if not ret:
            break
        start_time = time.time()
        if args.is_resize:
            img_raw = cv2.resize(img_raw, (args.resize, args.resize),interpolation=cv2.INTER_LINEAR)
        img = img_raw.copy()#cv2.cvtColor(img_raw.copy(),cv2.COLOR_BGR2RGB)
        try:
            det,patch = process(img,detector, output_size=(112, 112))
            for i,b in enumerate(det):
                detect_score = b[4]
                b = list(map(int, b))
                cx = b[0]
                cy = b[1] + 12
                cy2 = b[1] 
                cy3 = b[1] - 12
                cy4 = b[1] - 24
                
                p = cv2.resize(patch[i],(128,128))
                sim,idx = distinct_face(arc_model,Faces.feats,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
                
                draw_name = Faces.names[idx]
                if args.indivisual_threshold:
                    if detect_score < opt.detect_threshold:
                        continue
                    if sim < opt.sim_threshold:
                        draw_name = '???'
                        sim = -1.
                else:
                    if detect_score* sim < opt.detect_threshold * opt.sim_threshold:#sim < 0.15 :
                        draw_name = '???'
                        sim = -1.
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
        except:
            pass
        cv2.putText(img_raw, 'Save feature of face, Press "s"', (0, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
        cv2.imshow('result',img_raw)
        if args.write:
            out.write(cv2.resize(img_raw, (args.write_size, args.write_size),interpolation=cv2.INTER_LINEAR))
        k=cv2.waitKey(1)
        if k==115:
            is_ok = False
            while not is_ok:
                number = int(input("input feature number : "))
                p = cv2.cvtColor(cv2.resize(patch[number],(128,128)),cv2.COLOR_BGR2RGB)
                plt.imshow(p)
                plt.show()
                answer = input("is it right? (Y/N) : ")
                if answer=='Y':
                    name = input("input feature name : ")
                    feat = get_face_feature(arc_model,cv2.cvtColor(p,cv2.COLOR_BGR2GRAY))
                    save_feat(opt,name,feat,Faces)
                    answer = input("Will you keep saving it? (Y/N)")
                    if answer=='N':
                        is_ok=True
        elif k>0:
            break
        print("all_time",time.time()-start_time,'\n')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Arc face & Retina face inference')
    # input
    parser.add_argument('--img_mode', action ='store_true', help='Default is false (video mode), If you want to inference for img use --img_mode ')
    parser.add_argument('--src_path', type=str, help = 'Path of input data', default='')
    parser.add_argument('--is_resize', action ='store_true', help='Default is false, If you want to resize input image, use --is_resize')
    parser.add_argument('--resize', type=int, default=416, help='Size of resized image')

    # save
    parser.add_argument('--write',action='store_true', help='Default is false,If you want to save result, use --write')
    parser.add_argument('--write_size', type=int, default=416, help='Image or Frame size of result')
    parser.add_argument('--result_name', type=str, help = 'Name of result file. Please set --write', default='result')

    # threshold   
    parser.add_argument('--indivisual_threshold',action='store_true', help='Default is false, If you use this flag, confidence scroe and similarity threshold will be applied indivisually')
    args = parser.parse_args()

    opt = Config()
    arc_model = resnet_face18(opt.use_se)
    load_model(arc_model, opt.test_model_path)
    
    cudnn.benchmark = True
    device = torch.device("cuda")
    arc_model.to(device)
    arc_model.eval()
    detector = RetinafaceDetector(is_gpu=False,opt=opt)
    Faces = FaceFeatures(opt.features_path)

    if args.img_mode:
        img_mode(opt,args,arc_model,detector,Faces)
    else:
        video_mode(opt,args,arc_model,detector,Faces)
 


