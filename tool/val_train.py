# -*- coding: utf-8 -*-
import time
import os, sys, math
import argparse
from collections import deque
import datetime
import xml.etree.ElementTree as ET
import shutil

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image

from tool.utils import *
from tool.torch_utils import *
from tool.get_xmatrix import get_matrix
from tool.utils_map import get_map,get_coco_map



def val(model, device, config , epoch):
    MINOVERLAP = 0.5    #map iou
    use_cuda = 1
    #是否绘制中间结果的ap图
    draw_plot = False
    dataset=config.dataset
    #   结果输出的文件夹，默认为map_out
    ###################################################
    if epoch+1==config.TRAIN_EPOCHS:        
        draw_plot = True
    map_out_path    = os.path.join(config.name,'map_out')
    images_dir = os.path.join(config.dataset,'JPEGImages')
    val_path = os.path.join(config.dataset,'ImageSets','test.txt')

    
    image_ids=open(val_path).read().strip().split()
    #print(image_ids)
    
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    ###########################################################3
    num_classes = config.classes
    namesfile = os.path.join('data',dataset+'.names')#'data/custom.names'
    class_names = load_class_names(namesfile)
    
    print("Get predict result.")
    pure_inf_time = 0
    fps = 0
    ############################################################
    for num,image_id in enumerate(image_ids): 
        image_path  = os.path.join(images_dir,image_id+".jpg")
        img = cv2.imread(image_path)
        sized = cv2.resize(img, (config.width, config.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        ######################
        #start = time.time()
        #######################双路
        if num==0:
            for i in range(2):
                boxes, edge_pred,elapsed= do_detect_bgnet(model, sized, 0.1, 0.6, use_cuda)
                #boxes = do_detect(model, sized, 0.1, 0.6, use_cuda)#[bs,n_class*k,4+1+1+1]
                if i==1:
                    pure_inf_time += elapsed
        else:
            boxes, edge_pred,elapsed= do_detect_bgnet(model, sized, 0.1, 0.6, use_cuda)
            pure_inf_time += elapsed
        if draw_plot:
            savedir=os.path.join(config.name,'result')   #save pred images
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            savename=os.path.join(savedir,str(image_id)+".jpg")
            plot_boxes_cv2(img, boxes[0], savename=savename, class_names=class_names)
            temp_path=os.path.join(config.name,'edge_pred')
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            if isinstance(edge_pred,list):
                save_image(edge_pred[-1], os.path.join(temp_path,str(image_id)+'.jpg'))
            elif edge_pred.shape[1]!=1:
                save_image(torch.split(edge_pred,edge_pred.shape[1],1)[0], os.path.join(temp_path,str(image_id)+'.jpg'))
            else:
                save_image(edge_pred, os.path.join(temp_path,str(image_id)+'.jpg'))
        fps = (num + 1 ) / pure_inf_time
        get_map_txt(img,boxes[0],map_out_path,class_names,image_id)            
    print("Get predict result done.")
    ###################################################################
    print("Get ground truth result.")
    Annotation_path=os.path.join(config.dataset,'Annotations')
    for image_id in image_ids:
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join( Annotation_path,image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")
    
    screen_dir = os.path.join(name,config.weight.split('.')[0]+'.txt')
    screen_file=open(screen_dir,mode="a",encoding="utf-8")
    print("epoch : "+str(epoch+1),file=screen_file)
    print(f"fps : {fps:.1f} img / s",file=screen_file)
    print("Get map.",file=screen_file)
    screen_file.close()
    get_map(MINOVERLAP, draw_plot, screen_dir,path = map_out_path)
    screen_file=open(screen_dir,mode="a",encoding="utf-8")
    print("Get map done.",file=screen_file)

    print("Get coco map.",file=screen_file)
    screen_file.close()
    get_coco_map(class_names, map_out_path, screen_dir)
    screen_file=open(screen_dir,mode="a",encoding="utf-8")
    print("Get coco map done.",file=screen_file)
    screen_file.close()
    

def get_map_txt(img,boxes,map_out_path,class_names,image_id):
    img = np.copy(img)
    f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box=boxes[i]   #[4+1+1+1]--box+conf+conf+cls+id
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        score=box[5]
        pre_class=class_names[int(box[6])]

        f.write("%s %s %s %s %s %s\n"%(pre_class,score,str(x1),str(y1),str(x2),str(y2)))
    f.close()
    return


