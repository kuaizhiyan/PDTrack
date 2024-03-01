# 骨架网络构建
# from mmengine.registry import Registry
import torch
import numpy as np
from mmdet.registry import MODELS
from mmengine.config import Config
from mmdet.models.necks import ChannelMapper

if __name__ == '__main__':
    
    """
    Scale:
    
    input(x) : [b,3,256,128]
                |
           +--------+ 
           |backbone|
           +--------+
                |
                V
           ([b,256,64,32],[b,512,32,16],
            [b,1024,16,8], [b,2048,8,4])
                |
                V
             +----+     
             |neck|
             +----+
                |
                V
            [b,2048,8,4]
                |
                V
              +----+  
              |head|
              +----+
                |
                V
          [b,num_classes] --> classfiy
    
    """
    
    cfg = Config.fromfile('/home/kzy/project/mmdetection/PartQuerier/configs/testmodel.py')
    
    model = MODELS.build(cfg.model)
    
    x = [torch.randn(32,2048,8,4)] # 16*8=128
    
    out = model(x)
    print(out)
    print(out[0].shape)