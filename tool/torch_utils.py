import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils 


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda, nms_kind='nms'):
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
        else:
            print("unknow image type")
            exit(-1)

        
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        if nms_kind == 'diounms':
            return utils.post_processing_diou(img, conf_thresh, nms_thresh, output), t2-t1
        else:
            return utils.post_processing(img, conf_thresh, nms_thresh, output), t2-t1
            
def do_detect_bgnet(model, img, conf_thresh, nms_thresh, use_cuda):
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
        else:
            print("unknow image type")
            exit(-1)

        
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output, edge_output = model(img)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        return utils.post_processing(img, conf_thresh, nms_thresh, output),edge_output, t2-t1
        
def do_detect_bgnet_visualize(model, img, conf_thresh, nms_thresh, use_cuda,bgnet=False):
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
        else:
            print("unknow image type")
            exit(-1)

        
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        # t1 = time.time()
        if bgnet:
            output, edge_output,edge5,edge4,edge3 = model(img)
        else:
            output, edge_output,d2 = model(img)
        
        # t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        if bgnet:
            return utils.post_processing(img, conf_thresh, nms_thresh, output),edge_output,edge5,edge4,edge3
        else:
            return utils.post_processing(img, conf_thresh, nms_thresh, output),edge_output,d2
        
def do_detect_double(model, img,img1, conf_thresh, nms_thresh, use_cuda):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        else:
            print("unknow image type")
            exit(-1)

        img=torch.cat([img,img1],dim=0)
        if use_cuda:
            img = img.cuda()
        
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        return utils.post_processing(img, conf_thresh, nms_thresh, output), t2-t1
        
def do_detect_edge_direct(model, img,img1, conf_thresh, nms_thresh, use_cuda):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        else:
            print("unknow image type")
            exit(-1)

        img=torch.cat([img,img1],dim=1)
        if use_cuda:
            img = img.cuda()
        
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        return utils.post_processing(img, conf_thresh, nms_thresh, output), t2-t1
        
def do_detect_nosame(model, img,img1, conf_thresh, nms_thresh, use_cuda,edge_pred=False):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        else:
            print("unknow image type")
            exit(-1)

        #img=torch.cat([img,img1],dim=0)
        if use_cuda:
            img = img.cuda()
            img1=img1.cuda()
        
        img = torch.autograd.Variable(img)
        img1 = torch.autograd.Variable(img1)

        t1 = time.time()
        if edge_pred:
            output,edge_output = model(img,img1)
        else:
            output = model(img,img1)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        if edge_pred:
            return utils.post_processing(img, conf_thresh, nms_thresh, output), edge_output, t2-t1
        else:
            return utils.post_processing(img, conf_thresh, nms_thresh, output), t2-t1

def do_detect_nosame_visualize(model, img,img1, conf_thresh, nms_thresh, use_cuda):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            #img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            #img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float()
            if type(img1) == np.ndarray and len(img1.shape) == 3:
                img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img1) == np.ndarray and len(img1.shape) == 4:
                img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)
        else:
            print("unknow image type")
            exit(-1)

        #img=torch.cat([img,img1],dim=0)
        if use_cuda:
            img = img.cuda()
            img1=img1.cuda()
        
        img = torch.autograd.Variable(img)
        img1 = torch.autograd.Variable(img1)

        t1 = time.time()
        
        output,d3_o,d3_e,d3 = model(img,img1)
        
        t2 = time.time()
        '''
        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')
        '''
        return utils.post_processing(img, conf_thresh, nms_thresh, output), d3_o,d3_e,d3
