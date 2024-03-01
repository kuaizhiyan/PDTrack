# 骨架网络构建
# from mmengine.registry import Registry
import torch
import numpy as np
from mmdet.registry import MODELS
from mmdet.models.necks import ChannelMapper

if __name__ == '__main__':
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,2,3), # 返回四个特征图的索引
        frozen_stages=1,       # 冻结stem和第一层
        # 表示全部的BN需要梯度更新
        norm_cfg=dict(type='BN', requires_grad=True), 
        norm_eval=True,       # 且全局网络的BN进入eval模式
        style='pytorch')
    
    
    backbone=MODELS.build(backbone)
    neck=MODELS.build(dict(
        type='GlobalAveragePooling',
        kernel_size=(8, 4), stride=1))
    head=MODELS.build(dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=2048,    # 2048
        fc_channels=1024,
        out_channels=128,
        num_classes=380,            ### 如何确定？？ 
        loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')))
      
    x = torch.randn(2,3,256,128)
    
    all_backbone_outputs = backbone(x)
    
    channel_mapper = ChannelMapper([512,1024,2048],out_channels=256)
    
    print(type(all_backbone_outputs))
    print(len(all_backbone_outputs))
    for out in all_backbone_outputs:
        print(out.shape)
    
    # out = channel_mapper(out)
    # print(type(out))
    
    # backbone_outputs = (all_backbone_outputs[-1],) # [2,2048,8,4]
    
    # neck_outputs = neck(backbone_outputs)   # [2,2048] 全部展开
    # print(neck_outputs[0].shape)
    
    # # 经过 Neck(GAP) 后输出的向量维度应与 head.in_channel 一致
    # head_outputs = head.forward(neck_outputs)
    # print(head_outputs.shape)
    
    

