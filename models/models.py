import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
from models.bgnet import *

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class SkipConnect(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.kv1 = nn.Linear(channel, channel * 2)
        self.kv2 = nn.Linear(channel, channel * 2)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=1)
        self.conv = Conv_Bn_Activation(channel,channel,3,1,"relu")
    
    def forward(self,x_up,x_skip):
        up1, up2 = torch.chunk(x_up,2,dim=1) 
        skip1, skip2 = torch.chunk(x_skip,2,dim=1)
        b,c,h,w = up1.shape
        
        # B,N,C
        shortcut1 = up1+skip1
        up1 = up1.flatten(2).transpose(-2, -1)
        skip1 = skip1.flatten(2).transpose(-2, -1)      
        k1, v1 = torch.chunk(self.kv1(skip1),2,dim=2)        
        sout = (self.softmax1(up1@k1.transpose(-2,-1))@v1).transpose(-2,-1).reshape(b,c,h,w)+shortcut1
        
        shortcut2 = up2+skip2
        up2 = up2.flatten(2).transpose(-2, -1)
        skip2 = skip2.flatten(2).transpose(-2, -1)
        k2, v2 = torch.chunk(self.kv2(skip2),2,dim=2)
        cout = (v2@self.softmax2(up2.transpose(-2,-1)@k2)).transpose(-2,-1).reshape(b,c,h,w)+shortcut2
        
        out = self.conv(torch.cat([sout,cout],dim=1))
        
        return out

class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

# resnet18
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, base_width = 64):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#resnet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_out = self.conv1(x)
        x = self.bn1(x_out)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x_out,x2,x3,x4,x5
        

class Neck(nn.Module):
    def __init__(self, backbone, inference=False):
        super().__init__()
        self.inference = inference
        if 'resnet50' in backbone:
            self.conv1 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        elif 'resnet18' in backbone:
            self.conv1 = Conv_Bn_Activation(512, 512, 1, 1, 'leaky')
        else:
            self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        #print(backbone)
        if 'resnet50' in backbone:
            #print('resnet50')
            self.conv8 = Conv_Bn_Activation(1024, 256, 1, 1, 'leaky')
        elif 'resnet18' in backbone:
            self.conv8 = Conv_Bn_Activation(256, 256, 1, 1, 'leaky')
        else:
            self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        if 'resnet50' in backbone:
            self.conv15 = Conv_Bn_Activation(512, 128, 1, 1, 'leaky')
        elif 'resnet18' in backbone:
            self.conv15 = Conv_Bn_Activation(128, 128, 1, 1, 'leaky')
        else:
            self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference
        
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, backbone='darknet', n_classes=6, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3
        self.backbone = backbone
        # backbone
        if self.backbone == 'darknet' or self.backbone == 'darknet-neck':
            self.down1 = DownSample1()
            self.down2 = DownSample2()
            self.down3 = DownSample3()
            self.down4 = DownSample4()
            self.down5 = DownSample5()
        elif 'resnet50' in self.backbone:
            layers=[3,4,6,3]
            self.down=ResNet(Bottleneck,layers)
        elif 'resnet101' in self.backbone:
            layers=[3,4,23,3]
            self.down=ResNet(Bottleneck,layers)
        elif 'resnet18' in self.backbone:
            layers=[2,2,2,2]
            self.down=ResNet(BasicBlock,layers)

        
        if 'resnet50' in self.backbone:
            self.eam = EGM_each_ms_scaleadd_uplarge()
        elif 'resnet18' in self.backbone:
            self.eam = EGM_each_ms_scaleadd_uplarge(backbone='resnet18')
        elif 'darknet' in self.backbone:
            self.eam = EGM_each_ms_scaleadd_uplarge_darknet()

        
        # neck
        self.neck = Neck(self.backbone, inference)
        # yolov4conv137
        if yolov4conv137weight:
            if  backbone == 'darknet-neck':
                _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            elif backbone == 'darknet':
                _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5)
            else:
                _model = nn.Sequential(self.down)
            
            if 'sketch' in backbone:
                pretrained_dict = torch.load(yolov4conv137weight,map_location='cuda:0')['state_dict']
                
            else:
                pretrained_dict = torch.load(yolov4conv137weight)
            
                
            #print(pretrained_dict)
            #主要针对resnet-imagenet
            if 'imagenet' in backbone:
                model_dict = _model.state_dict()
                temp_dict = _model.state_dict()
            
                for n in list(temp_dict.keys()):
                    if 'num_batches_tracked' in n:
                        del temp_dict[n]
                # 1. filter out unnecessary keys     
                pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), temp_dict)}
            else:
              model_dict = _model.state_dict()
              # 1. filter out unnecessary keys
              #
              #pytorch0.4及以后版本
              #for (k, v), k1 in zip(pretrained_dict.items(), model_dict):
                #print(k,v.shape,model_dict[k1].shape)
              pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
              
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
            if 'imagenet' in backbone:
                del temp_dict
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)
        


    def forward(self, input):   
        if self.backbone == 'darknet' or self.backbone == 'darknet-neck':
            d1 = self.down1(input)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
        else:
            d1,d2,d3,d4,d5=self.down(input)
            
        edge = self.eam(d5,d4,d3,d2,d1)
        edge_att = torch.sigmoid(edge)
        
        x20, x13, x6 = self.neck(d5, d4, d3)
        output = self.head(x20, x13, x6)
        oe = F.interpolate(edge_att, scale_factor=2, mode='bilinear', align_corners=False)
        return output,oe


