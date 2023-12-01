# -*- coding: utf-8 -*-
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

Cfg.batch = 64
Cfg.subdivisions = 16
'''
number of mini_batches in one batch, size mini_batch = batch/subdivisions,
so GPU processes mini_batch samples at once,
and the weights will be updated for batch samples (1 iteration processes batch images)
'''
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.9
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.01
Cfg.burn_in = 1000

Cfg.steps = [3120, 3510]

Cfg.policy = Cfg.steps
Cfg.scales = .1, .1
Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 34
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 300
Cfg.TRAIN_OPTIMIZER = 'sgd'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'ciou'  # 'giou', 'diou', 'ciou'计算loss

Cfg.keep_checkpoint_max = 17
