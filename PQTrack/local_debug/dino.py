import torch
import numpy as np
from itertools import repeat
import collections.abc
from mmdet.registry import MODELS
from mmengine.config import Config


if __name__ == '__main__':
    
    cfg = Config.fromfile('/home/kzy/project/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py')
    
    backbone = MODELS.build(cfg.model.backbone)
    
    neck = MODELS.build(cfg=cfg.model.neck)
    
    encoder = MODELS.build(cfg.model.encoder)
    
    x = torch.randn(2,3,256,128)
    
    out = backbone(x)
    print(out)