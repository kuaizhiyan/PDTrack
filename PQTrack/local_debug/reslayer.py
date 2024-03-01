# 骨架网络构建
# from mmengine.registry import Registry
import torch
import numpy as np
from mmdet.registry import MODELS
from mmdet.models.layers import ResLayer,SimplifiedBasicBlock
from mmdet.models.necks import ChannelMapper
from mmdet.models.backbones.resnet import Bottleneck
from mmcv.cnn import ConvModule
import torch.nn as nn

if __name__=='__main__':
    # modellist = []
    # modellist.append(ConvModule(in_channels=16,out_channels=1024,kernel_size=1,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')))
    # modellist.append(ResLayer(block=SimplifiedBasicBlock,inplanes=1024,planes=1024,num_blocks=1))
    # modellist.append(ConvModule(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,stride=2) )
    
    # modellist.append(ConvModule(in_channels=1024,out_channels=2048,kernel_size=1,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')) )
    # modellist.append(ResLayer(block=SimplifiedBasicBlock,inplanes=2048,planes=2048,num_blocks=1))
    # modellist.append(ConvModule(in_channels=2048,out_channels=2048,kernel_size=3,padding=1,stride=2,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')))
    
    # x = torch.randn(2,16,32,16)
    
    # for i,layer in enumerate(modellist):
    #     x = layer(x)
    #     print(x.shape)
    channel_mapper = ChannelMapper([2048],256,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU'))
    x = torch.randn([1,2048,8,4])
    out = channel_mapper([x])
    print(out.shape)
    