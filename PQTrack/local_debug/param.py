import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.analysis import get_model_complexity_info
from torchsummary import summary

if __name__ == '__main__':
    input_shape = (1,3,256,128)
    
    cfg = Config.fromfile('/home/kzy/project/mmdetection/configs/strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py')
    model = MODELS.build(cfg=cfg.model)
    
    summary(model,input_shape)
    
    # analysis_results = get_model_complexity_info(model=model,input_shape=input_shape)
    # print(analysis_results)