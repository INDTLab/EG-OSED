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
from models.models import Yolov4
from tool.get_xmatrix import get_matrix
from tool.utils_map import get_map,get_coco_map



def test(model, device, config):
    
    use_cuda=1
    map_out_path    = os.path.join(config.name,'map_out')  #map_out_path
    draw_plot=True
    
    savedir=os.path.join(config.name,'result')   #save pred images
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    images_dir = os.path.join('dataset',config.dataset,'JPEGImages')
    test_path = os.path.join('dataset',config.dataset,'ImageSets','test.txt')
    image_ids=open(test_path).read().strip().split()
    #print(image_ids)
    
        
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
        
    num_classes = config.classes
    namesfile = os.path.join('data',config.dataset+'.names')  # custom class file--data/FOSD_OD.names
    class_names = load_class_names(namesfile)
    
    print("Get predict result.")
    
    pure_inf_time = 0
    fps = 0
    for num,image_id in enumerate(image_ids):           
        savename=os.path.join(savedir,str(image_id)+".jpg")
        image_path  = os.path.join(images_dir,image_id+".jpg")
        #print(image_path)
        img = cv2.imread(image_path)
        sized = cv2.resize(img, (config.width, config.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        if num==0:
            for i in range(2):
                start = time.time()
                boxes, edge_pred,elapsed= do_detect_bgnet(model, sized, 0.1, 0.6, use_cuda)#[bs,n_class*k,4+1+1+1]
                #boxes,elapsed= do_detect(model, sized, 0.1, 0.6, use_cuda)#[bs,n_class*k,4+1+1+1]
                #def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1)
                finish = time.time()
                #elapsed = finish - start
                if i == 1:
                #######################################################
                    pure_inf_time += elapsed
                    #print('%s: Predicted in %f seconds.' % (image_id+".jpg", (finish - start)))
        else:
            start = time.time()
            boxes, edge_pred,elapsed= do_detect_bgnet(model, sized, 0.1, 0.6, use_cuda)#nms conf和iou
            #boxes,elapsed= do_detect(model, sized, 0.1, 0.6, use_cuda)#[bs,n_class*k,4+1+1+1]
            finish = time.time()
            #elapsed = finish - start
            pure_inf_time += elapsed
            #print('%s: Predicted in %f seconds.' % (image_id+".jpg", (finish - start)))
        
        '''
        edge_pred_path=os.path.join(config.name,'edge_pred')
        
        if not os.path.exists(edge_pred_path):
            os.makedirs(edge_pred_path)
        if isinstance(edge_pred,list):
            save_image(edge_pred[0], os.path.join(edge_pred_path,str(image_id)+'.jpg'))
        else:
            save_image(edge_pred, os.path.join(edge_pred_path,str(image_id)+'.jpg'))
        '''
        fps = (num + 1 ) / pure_inf_time
        plot_boxes_cv2(img, boxes[0], savename=savename, class_names=class_names)
        get_map_txt(img,boxes[0],map_out_path,class_names,image_id)            
    print("Get predict result done.")
    ###################################################################
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join('dataset',config.dataset,"Annotations",image_id+".xml")).getroot()
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

    
    screen_path = os.path.join(config.name,'result.txt')
    screen_file=open(screen_path,mode="w",encoding="utf-8")
    print(f"fps : {fps:.1f} img / s",file=screen_file)
    print("Get map.",file=screen_file)
    screen_file.close()
    get_map(MINOVERLAP, draw_plot, screen_path, path = map_out_path)
    screen_file=open(screen_path,mode="a",encoding="utf-8")
    print("Get map done.",file=screen_file)
    '''
    print("Get coco map.",file=screen_file)
    screen_file.close()
    get_coco_map(class_names, map_out_path, screen_path)
    screen_file=open(screen_path,mode="a",encoding="utf-8")
    print("Get coco map done." ,file=screen_file)
    screen_file.close()
    '''

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

###############################################
def get_args():
    
    parser = argparse.ArgumentParser(description='Test your image or video by trained model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-name',  type=str, default='fosd-egm-param',help='save name', dest='name')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1', help='GPU', dest='gpu')
    parser.add_argument('-dataset', type=str, default='FOSD_OD', help='dataset', dest='dataset')
    parser.add_argument('-w','--weight', type=str, default='/data2/xkk/yolo4/checkpoint/fosd/newstep/dual/fosd_pres50_dual_eam_eachscpcuplarge_param_loss11_nesterov/Yolov4_epoch300.pth', help='Yolov4_epoch300.pth')
    parser.add_argument('-backbone',type=str, default='resnet50', help='backbone')
    parser.add_argument('--width', type=int, default='608',help='image width')
    parser.add_argument('--height', type=int, default='608',help='image height')
    args =parser.parse_args()

    return args




def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    
    cfg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.dataset == 'FOSD_OD':
        cfg.classes = 34
    elif cfg.dataset == 'Places365_OD':
        cfg.classes = 40
    elif cfg.dataset == 'SUN_OD':
        cfg.classes = 6

    cfg.name = os.path.join('runs/train',cfg.name)

    MINOVERLAP      = 0.5    #map iou
    
    model = Yolov4(yolov4conv137weight=None, backbone= cfg.backbone, n_classes=cfg.classes, inference=True)
    print("Load model.")
    pretrained_dict = torch.load(cfg.weight, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    print('Loading weights from %s... Done!' % (cfg.weight))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        test(model=model,
              config=cfg,
              device=device, )
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
