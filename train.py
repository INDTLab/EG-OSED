# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import os
seed=666
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import time
import logging
import sys, math
import argparse
from collections import deque
import datetime
from torchvision.utils import make_grid, save_image

import cv2
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict


from tool.dataset import Yolo_dataset
from cfg import Cfg
from models.models import Yolov4

from tool.tv_reference.utils import collate_fn as val_collate
from tool.val_train import val


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = torch.tensor([all_anchors_grid[j] for j in self.anch_masks[i]], device=self.device).float()
            ref_anchors = torch.zeros((len(all_anchors_grid), 4), device=self.device).float()
            ref_anchors[:, 2:] = torch.tensor(all_anchors_grid, device=self.device).float()
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = masked_anchors[:, 0].repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2)
            anchor_h = masked_anchors[:, 1].repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16)
        truth_j_all = truth_y_all.to(torch.int16)

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            #print(pred[b].shape)
            pred_ious = bboxes_iou(pred[b].contiguous().view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16)] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
    
            pred = output[..., :4].clone()
            #print(pred.shape)
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2
        
def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)#bs,c,h,w
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=10, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label,config.dataset_dir, config.edge_dir, config, train=True)

    n_train = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)


    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')


    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:      {config.pretrained}
        train_steps:     {config.steps[0],config.steps[1]}
    ''')#

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            [{'params':model.parameters(),
            'initial_lr': config.learning_rate/ config.batch}],            
            lr=config.learning_rate/ config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(), 
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
            nesterov=True,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)

    keep_checkpoint_max = config.keep_checkpoint_max
    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    global_step = 0
    model.train()
    for epoch in range(0,epochs):

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=60) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                images = batch[0]
                bboxes = batch[1]
                
                images,edges = torch.split(images, [3,1], dim=1)
                '''
                temp_path='temp_batch'
                if not os.path.exists(temp_path):
                    os.mkdir(temp_path)
                images0=images
                images0=make_grid(images0,2,0)
                save_image(images0, os.path.join(temp_path,str(i)+'.jpg'))
                
                images0=images[0]
                
                print(images0.shape)
                images1=edges
                images1=make_grid(images1,2,0)
                save_image(images1, os.path.join(temp_path,str(i)+'_1.jpg'))
                
                images1=images[1]
                #print(images1.shape)
                '''
                
                images=images.type(torch.cuda.FloatTensor)
                images = images.to(device=device)
                edges=edges.type(torch.cuda.FloatTensor)
                edges = edges.to(device=device)
                bboxes = bboxes.to(device=device)

                bboxes_pred, edge_pred  = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss_edge = dice_loss(edge_pred, edges)
                loss=loss+loss_edge
                # loss = loss / config.subdivisions
                loss.backward()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_edge', loss_edge.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_last_lr()[0] * config.batch, global_step)
                    
                    logging.debug('Train step_{}: loss : {},loss_edge {} ,loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_edge.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_last_lr()[0] * config.batch))

                pbar.update(images.shape[0])
                
            if save_cp:
                try:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                
                if isinstance(model, torch.nn.DataParallel):
                   torch.save(model.module.state_dict(), save_path)
                else:
                   torch.save(model.state_dict(), save_path)                
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > keep_checkpoint_max > 0:                     
                    keep_list=[150,200,250,260,270,280,290]
                    model_to_remove = saved_models.popleft()
                    pthname=model_to_remove.split('/')[-1]
                    pthnum=pthname.split('.pth')[0].split('epoch')[-1]
                    if int(pthnum) in keep_list or int(pthnum)<=300 and int(pthnum)>=291:
                        model_to_remove = saved_models.popleft()
                        keep_checkpoint_max-=1
                    try:                         
                        os.remove(model_to_remove)
                    except:                         
                        logging.info(f'failed to remove {model_to_remove}')
            
            if (epoch+1)>=150 and (epoch+1)%50 == 0 or (epoch+1)>=250 and (epoch+1)%10 == 0 or epoch==0:    
                
                eval_model = Yolov4(config.pretrained,backbone=config.backbone, n_classes=config.classes, inference=True)
            
                if torch.cuda.device_count() > 1:
                    #checkpoint={k:v for k,v in model.module.state_dict().items() if 'total' not in k}
                    eval_model.load_state_dict(model.module.state_dict())
                else:
                    #checkpoint={k:v for k,v in model.state_dict().items() if 'total' not in k}
                    eval_model.load_state_dict(model.state_dict())
                eval_model.to(device)
                eval_model.eval()
                with torch.no_grad():
                     val(eval_model,device,config,epoch)
                del eval_model

            

    writer.close()


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sub', '--subdivisions', type=int, default=8,help='subdivisions', dest='subdivisions')
    parser.add_argument('-name',  type=str, default='fosd-egm-param',help='project name ', dest='name')
    parser.add_argument('-dataset',  type=str, default='FOSD_OD',help='dataset', dest='dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',help='GPU', dest='gpu')
    parser.add_argument('-backbone', type=str, default='resnet50', help='darknet\darknet-neck\resnet18-imagenet\resnet50')
    parser.add_argument('-pretrained', type=str, default='weights/resnet50_places365.pt',help='pretrained yolov4.conv.137')
    parser.add_argument(
        '-iou-type', type=str, default='ciou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=17,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    if cfg.dataset == 'FOSD_OD':
        cfg.steps=[8805,9906]
        cfg.classes = 34
    elif cfg.dataset == 'Places365_OD':
        cfg.steps=[14550,16369]
        cfg.classes = 40
    elif cfg.dataset == 'SUN_OD':
        cfg.steps=[2895,3257]
        cfg.classes = 6

    cfg.name = os.path.join('runs/train',cfg.name)
    cfg.dataset_dir = os.path.join('dataset',cfg.dataset,'JPEGImages')
    cfg.edge_dir = os.path.join('dataset',cfg.dataset,'EdgeMap-canny')
    cfg.train_label = os.path.join('data',cfg.dataset,'train.txt')   
    cfg.val_path = os.path.join('dataset',cfg.dataset,'ImageSets/test.txt')
    cfg.checkpoints = os.path.join(cfg.name,'checkpoints')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Yolov4(cfg.pretrained,backbone=cfg.backbone, n_classes=cfg.classes)

    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device=device)
    

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
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
