import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import log
from functools import partial


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes=64, planes=64, dilation=[1,2,4,8], baseWidth=16, scale=4, aggregation=True, use_dwconv=False):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        #self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            if use_dwconv:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], groups=self.width, bias=False))
            else:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.aggregation = aggregation

    def forward(self, x):
        #out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        spx = torch.split(x, self.width, 1)
        out = []
        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i==0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out
        

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
        

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# direct_add
class EGM_each_ms_scaleadd_uplarge(nn.Module):
    def __init__(self,ms='scpc',scale=None,backbone='resnet50'):
        super(EGM_each_ms_scaleadd_uplarge, self).__init__()
        if scale == None:
            scale = nn.Parameter(torch.ones(5))        
        self.scale = scale
        self.ms = ms
        if backbone == 'resnet50':
            oc=[64,256,512,1024,2048]
            nc=[64,64,128,256,512]
        elif backbone == 'resnet18':
            oc=[64,64,128,256,512]
            nc=[64,64,64,64,128]
        #self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(oc[1], nc[1], 1)
        self.reduce3 = nn.Conv2d(oc[2], nc[2], 1)
        self.reduce4 = nn.Conv2d(oc[3], nc[3], 1)
        self.reduce5 = nn.Conv2d(oc[4], nc[4], 1)
        
        self.getf1 = ReceptiveConv(nc[0], nc[0], [1,2,4,8])
        self.getf2 = ReceptiveConv(nc[1], nc[1], [1,2,4,8])
        self.getf3 = ReceptiveConv(nc[2], nc[2], [1,2,4,8])
        self.getf4 = ReceptiveConv(nc[3], nc[3], [1,2,3,4])
        self.getf5 = ReceptiveConv(nc[4], nc[4], [1,2,3,4])
        
        self.reduce51 = nn.Conv2d(nc[4], nc[1], 1)
        self.reduce41 = nn.Conv2d(nc[3], nc[1], 1)
        self.reduce31 = nn.Conv2d(nc[2], nc[1], 1)
        self.reduce21 = nn.Conv2d(nc[1], nc[1], 1)
        
        
        self.conv = ConvBNR(nc[1],nc[1], 3)
        self.edge = nn.Conv2d(nc[1], 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):
        size = x1.size()[2:]
        
        x5_fea = self.getf5(self.reduce5(x5))
        x5 = self.reduce51(x5_fea)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        
        x4_fea = self.getf4(self.reduce4(x4)) 
        x4 = self.reduce41(x4_fea)      
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        
        x3_fea = self.getf3(self.reduce3(x3))     
        x3 = self.reduce31(x3_fea)   
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        
        x2_fea = self.getf2(self.reduce2(x2))
        x2_fea = self.reduce21(x2_fea)
        x2 = F.interpolate(x2_fea, size, mode='bilinear', align_corners=False)
        
        x1_fea = self.getf1(x1)
        x1 = self.conv(self.scale[0]*x1_fea+self.scale[1]*x2+self.scale[2]*x3+self.scale[3]*x4+self.scale[4]*x5)
        
        out = self.edge(x1)
        
        return out

# cascade_cat
class EGM_each_scpc(nn.Module):
    def __init__(self,):
        super(EGM_each_scpc, self).__init__()
        self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)
        
        self.scpc1 = ReceptiveConv(16, 16, [1,2,4,8])
        self.scpc2 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc3 = ReceptiveConv(128, 128, [1,2,4,8])
        self.scpc4 = ReceptiveConv(256, 256, [1,2,3,4])
        self.scpc5 = ReceptiveConv(512, 512, [1,2,3,4])
        
        # 64--16--64
        self.reduce12 = nn.Conv2d(16+64, 64, 1)
        self.reduce23 = nn.Conv2d(64+128, 64, 1)
        self.reduce34 = nn.Conv2d(256+128, 128, 1)
        self.reduce45 = nn.Conv2d(512+256, 256, 1)
        
        self.conv = ConvBNR(64, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):
        size1=x1.size()[2:]
        size2=x2.size()[2:]
        size3=x3.size()[2:]
        size4=x4.size()[2:]
        
        x5 = self.scpc5(self.reduce5(x5))
        x5 = F.interpolate(x5, size4, mode='bilinear', align_corners=False)
        
        x4 = self.scpc4(self.reduce4(x4))
        x4 = self.reduce45(torch.cat([x4,x5],dim=1))        
        
        x4 = F.interpolate(x4, size3, mode='bilinear', align_corners=False)
        
        x3 = self.scpc3(self.reduce3(x3))
        x3 = self.reduce34(torch.cat([x3,x4],dim=1))
        
        x3 = F.interpolate(x3, size2, mode='bilinear', align_corners=False)
        
        x2 = self.scpc2(self.reduce2(x2))
        x2 = self.reduce23(torch.cat([x2,x3],dim=1))
        
        x2 = F.interpolate(x2, size1, mode='bilinear', align_corners=False)
        
        x1 = self.scpc1(self.reduce1(x1))
        x1 = self.reduce12(torch.cat([x1,x2],dim=1))
        
        out_feature = self.conv(x1)
        out = self.edge(out_feature)
        
        return out

        
# direct_cat
class EGM_each_ms_cat_uplarge(nn.Module):
    def __init__(self,):
        
        super(EGM_each_ms_cat_uplarge, self).__init__()
        
        #self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)
        
        self.getf1 = ReceptiveConv(64, 64, [1,2,4,8])
        self.getf2 = ReceptiveConv(64, 64, [1,2,4,8])
        self.getf3 = ReceptiveConv(128, 128, [1,2,4,8])
        self.getf4 = ReceptiveConv(256, 256, [1,2,3,4])
        self.getf5 = ReceptiveConv(512, 512, [1,2,3,4])
        
        self.reduce51 = nn.Conv2d(512, 64, 1)
        self.reduce41 = nn.Conv2d(256, 64, 1)
        self.reduce31 = nn.Conv2d(128, 64, 1)
        #self.reduce21 = nn.Conv2d(64, 64, 1)
        
        
        self.conv = ConvBNR(64*5, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):
        size = x1.size()[2:]
        
        x5_fea = self.getf5(self.reduce5(x5))
        x5 = self.reduce51(x5_fea)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        
        x4_fea = self.getf4(self.reduce4(x4)) 
        x4 = self.reduce41(x4_fea)      
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        
        x3_fea = self.getf3(self.reduce3(x3))     
        x3 = self.reduce31(x3_fea)   
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        
        x2_fea = self.getf2(self.reduce2(x2))
        #x2 = self.reduce21(x2_fea)
        x2 = F.interpolate(x2_fea, size, mode='bilinear', align_corners=False)
        
        #x1 = self.scpc1(self.reduce1(x1))
        x1_fea = self.getf1(x1)
        x1 = self.conv(torch.cat((x1,x2,x3,x4,x5),dim=1))
        
        out = self.edge(x1)
        
        return out
        
    
# cascade_add
class EGM_each_scpc_scaleadd(nn.Module):
    def __init__(self,scale=None):
        super(EGM_each_scpc_scaleadd, self).__init__()
        if scale == None:
            scale = nn.Parameter(torch.ones(5))
        self.scale = scale
        
        #self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)
        
        self.scpc1 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc2 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc3 = ReceptiveConv(128, 128, [1,2,4,8])
        self.scpc4 = ReceptiveConv(256, 256, [1,2,3,4])
        self.scpc5 = ReceptiveConv(512, 512, [1,2,3,4])
        
        self.reduce54 = nn.Conv2d(512, 256, 1)
        self.reduce43 = nn.Conv2d(256, 128, 1)
        self.reduce32 = nn.Conv2d(128, 64, 1)
        self.conv12 = nn.Conv2d(64, 64, 1)
        
        
        self.conv = ConvBNR(64, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):
        size1=x1.size()[2:]
        size2=x2.size()[2:]
        size3=x3.size()[2:]
        size4=x4.size()[2:]
        
        x5 = self.reduce54(self.scpc5(self.reduce5(x5)))
        x5 = F.interpolate(x5, size4, mode='bilinear', align_corners=False)
        
        x4 = self.scpc4(self.reduce4(x4))
        x4 = self.reduce43(x4+self.scale[4]*x5)
        
        x4 = F.interpolate(x4, size3, mode='bilinear', align_corners=False)
        
        x3 = self.scpc3(self.reduce3(x3))
        x3 = self.reduce32(x3+self.scale[3]*x4)
        
        x3 = F.interpolate(x3, size2, mode='bilinear', align_corners=False)
        
        x2 = self.scpc2(self.reduce2(x2))
        x2 = self.conv12(x2+self.scale[2]*x3)
        
        x2 = F.interpolate(x2, size1, mode='bilinear', align_corners=False)
        
        #x1 = self.scpc1(self.reduce1(x1))
        x1 = self.scpc1(x1)
        x1 = self.conv(self.scale[0]*x1+self.scale[1]*x2)
        
        out = self.edge(x1)
        
        return out


# cascade_add     
class EGM_each_scpc_add(nn.Module):
    def __init__(self):
        super(EGM_each_scpc_add, self).__init__()
        #self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 64, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 256, 1)
        
        self.reduce42 = nn.Conv2d(256, 64, 1)
        
        self.scpc1 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc2 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc3 = ReceptiveConv(64, 64, [1,2,4,8])
        self.scpc4 = ReceptiveConv(256, 256, [1,2,3,4])
        self.scpc5 = ReceptiveConv(256, 256, [1,2,3,4])
        
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv23 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv45 = nn.Conv2d(256, 256, 3, 1, 1)
        
        #self.conv = ConvBNR(64, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
    def forward(self,x5,x4,x3,x2,x1):
        size1=x1.size()[2:]
        size2=x2.size()[2:]
        size3=x3.size()[2:]
        size4=x4.size()[2:]
        
        x5 = self.scpc5(self.reduce5(x5))
        x5 = F.interpolate(x5, size4, mode='bilinear', align_corners=False)
        
        x4 = self.scpc4(self.reduce4(x4))
        x4 = self.reduce42(self.conv45(x4+x5))
        
        x4 = F.interpolate(x4, size3, mode='bilinear', align_corners=False)
        
        x3 = self.scpc3(self.reduce3(x3))
        x3 = self.conv34(x3+x4)
        
        x3 = F.interpolate(x3, size2, mode='bilinear', align_corners=False)
        
        x2 = self.scpc2(self.reduce2(x2))
        x2 = self.conv23(x2+x3)
        
        x2 = F.interpolate(x2, size1, mode='bilinear', align_corners=False)
        
        #x1 = self.scpc1(self.reduce1(x1))
        x1 = self.scpc1(x1)
        x1 = self.conv12(x1+x2)
        
        #out_feature = self.conv(x1)
        #out = self.edge(out_feature)
        out = self.edge(x1)
        
        return out
    
class EGM_each_ms_scaleadd_uplarge_darknet(nn.Module):
    def __init__(self,scale=None):
        
        super(EGM_each_ms_scaleadd_uplarge_darknet, self).__init__()
        if scale == None:
            scale = nn.Parameter(torch.ones(5))    
        self.scale = scale
        #self.reduce1 = nn.Conv2d(64, 16, 1)
        self.reduce2 = nn.Conv2d(128, 64, 1)
        self.reduce3 = nn.Conv2d(256, 128, 1)
        self.reduce4 = nn.Conv2d(512, 256, 1)
        self.reduce5 = nn.Conv2d(1024, 512, 1)
        
        self.getf1 = ReceptiveConv(64, 64, [1,2,4,8])
        self.getf2 = ReceptiveConv(64, 64, [1,2,4,8])
        self.getf3 = ReceptiveConv(128, 128, [1,2,4,8])
        self.getf4 = ReceptiveConv(256, 256, [1,2,3,4])
        self.getf5 = ReceptiveConv(512, 512, [1,2,3,4])
        
        
        self.reduce51 = nn.Conv2d(512, 64, 1)
        self.reduce41 = nn.Conv2d(256, 64, 1)
        self.reduce31 = nn.Conv2d(128, 64, 1)
        self.reduce21 = nn.Conv2d(64, 64, 1)
        
        
        self.conv = ConvBNR(64, 64, 3)
        self.edge = nn.Conv2d(64, 1, 1)
        
        
    def forward(self,x5,x4,x3,x2,x1):
        size = x1.size()[2:]
        
        x5_fea = self.getf5(self.reduce5(x5))
        x5 = self.reduce51(x5_fea)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        
        x4_fea = self.getf4(self.reduce4(x4)) 
        x4 = self.reduce41(x4_fea)      
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        
        x3_fea = self.getf3(self.reduce3(x3))     
        x3 = self.reduce31(x3_fea)   
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)
        
        x2_fea = self.getf2(self.reduce2(x2))
        x2_fea = self.reduce21(x2_fea)
        x2 = F.interpolate(x2_fea, size, mode='bilinear', align_corners=False)
        
        x1_fea = self.getf1(x1)
        x1 = self.conv(self.scale[0]*x1_fea+self.scale[1]*x2+self.scale[2]*x3+self.scale[3]*x4+self.scale[4]*x5)
        
        out = self.edge(x1)
        return out
