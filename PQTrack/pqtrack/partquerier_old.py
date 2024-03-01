# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List,Dict,Optional,Union

import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from typing import List, Tuple, Union
from mmdet.models.detectors import ConditionalDETR
from mmdet.models.layers import (ConditionalDetrTransformerDecoder,
                      DetrTransformerEncoder, SinePositionalEncoding)
from mmdet.models.layers import ResLayer,SimplifiedBasicBlock
from mmdet.models.reid import GlobalAveragePooling
from mmcv.cnn import ConvModule

# # @MODELS.register_module()
# class PartQuerier(BaseModule):
#     """ Learn part features by part querier from feature maps.
    
#     Args:
#         num_queries: int the number of part query
    
#     """
    
#     def __init__(self,
#                  encoder: OptConfigType = None,
#                  decoder: OptConfigType = None,
#                  positional_encoding: OptConfigType = None,
#                  num_queries: int = 16,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                 #  part_querier_init_cfg:Optional[dict]=None
#                  ) -> None:
#         super(PartQuerier,self).__init__()
#         # process args
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
#         self.encoder = encoder      # cfg
#         self.decoder = decoder      # cfg
#         self.positional_encoding = positional_encoding  # cfg
#         self.num_queries = num_queries

#         # init model layers
#         self._init_layers()
    
#     def _init_layers(self) -> None:
#         """Initialize layers except for backbone, neck and bbox_head."""
#         self.positional_encoding = SinePositionalEncoding(
#             **self.positional_encoding)
#         self.encoder = DetrTransformerEncoder(**self.encoder)
#         self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
#         self.embed_dims = self.encoder.embed_dims
#         # NOTE The embed_dims is typically passed from the inside out.
#         # For example in DETR, The embed_dims is passed as
#         # self_attn -> the first encoder layer -> encoder -> detector.
#         self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

#         num_feats = self.positional_encoding.num_feats
#         assert num_feats * 2 == self.embed_dims, \
#             f'embed_dims should be exactly 2 times of num_feats. ' \
#             f'Found {self.embed_dims} and {num_feats}.'   
    
#     def init_weights(self) -> None:
#         """Initialize weights for Transformer and other components."""
#         super().init_weights()
#         for coder in self.encoder, self.decoder:
#             for p in coder.parameters():
#                 if p.dim() > 1:
#                     nn.init.xavier_uniform_(p)
    
#     def pre_transformer(
#             self,
#             img_feats: Tuple[Tensor],
#         ) -> Tuple[Dict, Dict]:
#         """Prepare the inputs of the Transformer.

#         The forward procedure of the transformer is defined as:
#         'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
#         More details can be found at `TransformerDetector.forward_transformer`
#         in `mmdet/detector/base_detr.py`.

#         Args:
#             img_feats (Tuple[Tensor]): Tuple of features output from the neck,
#                 has shape (bs, c, h, w).
#         Returns:
#             tuple[dict, dict]: The first dict contains the inputs of encoder
#             and the second dict contains the inputs of decoder.

#             - encoder_inputs_dict (dict): The keyword args dictionary of
#               `self.forward_encoder()`, which includes 'feat', 'feat_mask',
#               and 'feat_pos'.
#             - decoder_inputs_dict (dict): The keyword args dictionary of
#               `self.forward_decoder()`, which includes 'memory_mask',
#               and 'memory_pos'.
#         """

#         feat = img_feats[-1]  # NOTE img_feats contains only one feature.   # [4,3,32,16]
#         batch_size, feat_dim, h, w = feat.shape
        
#         # construct binary masks which for the transformer.    
#         masks = None
#         # [batch_size, embed_dim, h, w]
#         pos_embed = self.positional_encoding(masks, input=feat)
       
#         # use `view` instead of `flatten` for dynamically exporting to ONNX
#         # [bs, c, h, w] -> [bs, h*w, c]
#         feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
#         pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
#         # [bs, h, w] -> [bs, h*w]
#         if masks is not None:
#             masks = masks.view(batch_size, -1)

#         # prepare transformer_inputs_dict
#         encoder_inputs_dict = dict(
#             feat=feat, feat_mask=masks, feat_pos=pos_embed)
#         decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
#         return encoder_inputs_dict, decoder_inputs_dict  
    
#     def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
#                         feat_pos: Tensor) -> Dict:
#         """Forward with Transformer encoder.

#         The forward procedure of the transformer is defined as:
#         'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
#         More details can be found at `TransformerDetector.forward_transformer`
#         in `mmdet/detector/base_detr.py`.

#         Args:
#             feat (Tensor): Sequential features, has shape (bs, num_feat_points,
#                 dim).
#             feat_mask (Tensor): ByteTensor, the padding mask of the features,
#                 has shape (bs, num_feat_points).
#             feat_pos (Tensor): The positional embeddings of the features, has
#                 shape (bs, num_feat_points, dim).

#         Returns:
#             dict: The dictionary of encoder outputs, which includes the
#             `memory` of the encoder output.
#         """
#         memory = self.encoder(
#             query=feat, query_pos=feat_pos,
#             key_padding_mask=feat_mask)  # for self_attn
#         encoder_outputs_dict = dict(memory=memory)
#         return encoder_outputs_dict
    
#     def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
#         """Prepare intermediate variables before entering Transformer decoder,
#         such as `query`, `query_pos`.

#         The forward procedure of the transformer is defined as:
#         'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
#         More details can be found at `TransformerDetector.forward_transformer`
#         in `mmdet/detector/base_detr.py`.

#         Args:
#             memory (Tensor): The output embeddings of the Transformer encoder,
#                 has shape (bs, num_feat_points, dim).

#         Returns:
#             tuple[dict, dict]: The first dict contains the inputs of decoder
#             and the second dict contains the inputs of the bbox_head function.

#             - decoder_inputs_dict (dict): The keyword args dictionary of
#               `self.forward_decoder()`, which includes 'query', 'query_pos',
#               'memory'.
#             - head_inputs_dict (dict): The keyword args dictionary of the
#               bbox_head functions, which is usually empty, or includes
#               `enc_outputs_class` and `enc_outputs_class` when the detector
#               support 'two stage' or 'query selection' strategies.
#         """

#         batch_size = memory.size(0)  # (bs, num_feat_points, dim)
#         query_pos = self.query_embedding.weight
#         # (num_queries, dim) -> (bs, num_queries, dim)
#         query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
#         query = torch.zeros_like(query_pos)

#         decoder_inputs_dict = dict(
#             query_pos=query_pos, query=query, memory=memory)
#         head_inputs_dict = dict()
#         return decoder_inputs_dict, head_inputs_dict
    
#     def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
#                         memory_mask: Tensor, memory_pos: Tensor) -> Dict:
#         """Forward with Transformer decoder.

#         Args:
#             query (Tensor): The queries of decoder inputs, has shape
#                 (bs, num_queries, dim).
#             query_pos (Tensor): The positional queries of decoder inputs,
#                 has shape (bs, num_queries, dim).
#             memory (Tensor): The output embeddings of the Transformer encoder,
#                 has shape (bs, num_feat_points, dim).
#             memory_mask (Tensor): ByteTensor, the padding mask of the memory,
#                 has shape (bs, num_feat_points).
#             memory_pos (Tensor): The positional embeddings of memory, has
#                 shape (bs, num_feat_points, dim).

#         Returns:
#             dict: The dictionary of decoder outputs, which includes the
#             `hidden_states` and `references` of the decoder output.

#             - hidden_states (Tensor): Has shape
#                 (num_decoder_layers, bs, num_queries, dim)
#             - references (Tensor): Has shape
#                 (bs, num_queries, 2)
#         """

#         hidden_states, references = self.decoder(
#             query=query,
#             key=memory,
#             query_pos=query_pos,
#             key_pos=memory_pos,
#             key_padding_mask=memory_mask)
#         head_inputs_dict = dict(
#             hidden_states=hidden_states, references=references)
#         return head_inputs_dict
    
#     def forward(self,img_feats):    
        
#         encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
#             img_feats)  # img_feats tuple([1,256,25,38]) samples[..,img_shape=800,1199]
#         # encoder_inputs:[feat:[1,950,256],feat_mask:None,feat_pos[1,950,256]] decoder:
#         encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)  # :'memory'[1,950,256]

#         tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
#         decoder_inputs_dict.update(tmp_dec_in)  # {query_pos[1,300,256],query[1,300,256],memory[1,950,256]}
#         # decoder_inputs_dict {query,query_pos,memory,memory_pos}
#         decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)  #{hidden_state[6,1,300,256],reference[1,300,2]}
#         head_inputs_dict.update(decoder_outputs_dict)
        
#         return head_inputs_dict #dict{hidden_state[6,1,300,256],reference[1,300,2]}
        

# # @MODELS.register_module()
# class PartAggregator(BaseModule):
#     """
#     Remap the output query from PartQuerier to 2-dim [batch,num_query,h',w'] and downsample to
#     [batch,out_channels,h'/4,w'/4]. At last project it to 1-dim vector[batch,out_channels] as the 
#     input of classfier head.

#     Args:
#         bottleneck_nums: int the number of stacked bottlenecks
#         in_planes:int the output feature map dims from PartQuerier
#         inter_planes:int 
#         out_planes:int the output feature map dims of PartAggregator
#         agg_init_cfg: dict
    
#     Return:

#     """
#     def __init__(self,
#                  bottleneck_nums:int=2,
#                  in_planes:int=16,
#                  inter_planes:int=1024,
#                  out_planes:int=2048,
#                  ):
#         super(PartAggregator,self).__init__()
#         self.blocks = bottleneck_nums
#         self.in_planes = in_planes
#         self.inter_planes = inter_planes
#         self.out_planes = out_planes
#         self._init_layers()
    
#     def _init_layers(self):
#         modellist = nn.ModuleList()
#         # channel mapper
#         modellist.append(ConvModule(in_channels=self.in_planes,out_channels=self.inter_planes,kernel_size=1,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')))
#         # BottleNeck
#         modellist.append(ResLayer(block=SimplifiedBasicBlock,inplanes=self.inter_planes,planes=self.inter_planes,num_blocks=self.blocks))
#         # # downsample
#         # modellist.append(ConvModule(in_channels=self.inter_planes,out_channels=self.inter_planes,kernel_size=3,padding=1,stride=2))
#         # channel mapper
#         modellist.append(ConvModule(in_channels=self.inter_planes,out_channels=self.out_planes,kernel_size=1,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')))
#         # bottleneck
#         modellist.append(ResLayer(block=SimplifiedBasicBlock,inplanes=self.out_planes,planes=self.out_planes,num_blocks=self.blocks))
#         # downsample
#         modellist.append(ConvModule(in_channels=self.out_planes,out_channels=self.out_planes,kernel_size=3,padding=1,stride=2,norm_cfg=dict(type='BN'),act_cfg=dict(type='LeakyReLU')))
#         self.layers = modellist
#     def forward(self,x):
#         x = x[0]
#         for i,layer in enumerate(self.layers):
#             x = layer(x)
#         return x
        
    
# # @MODELS.register_module()
# class PartQuerier_neck(BaseModule):
#     """
#     The PartQuerier serves as the neck of the ReID network.
#     The data processing flow is as :
#          +-----------+
#          |PartQuerier|
#          +-----------+
#                |
#                V
#         +--------------+
#         |PartAggregator|
#         +--------------+
#                |
#                V
#        +----------------+ 
#        |GlobalAvgPooling|
#        +----------------+ 
    
#     Return:
#         out: tuple of outputs
    
#     """
    
#     def __init__(self, 
#                  afterpq_h:int=16,
#                  afterpq_w:int=16,
#                  batch_size=1,
#                  encoder: OptConfigType = None,
#                  decoder: OptConfigType = None,
#                  positional_encoding: OptConfigType = None,
#                  channel_mapper:OptConfigType = None,
#                  num_queries: int = 16,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                  aggregator: OptConfigType = None
#                  )->tuple:
#         super(PartQuerier_neck,self).__init__()
#         self.h = afterpq_h
#         self.w = afterpq_w
#         # self.batch = batch_size,
#         self.num_queries = num_queries
#         self.PartQuerier = PartQuerier(encoder=encoder,
#                                        decoder=decoder,
#                                        positional_encoding=positional_encoding,
#                                        num_queries=num_queries,
#                                        train_cfg=train_cfg,
#                                        test_cfg=test_cfg)
#         self.PartAggregator = PartAggregator(**aggregator)
#         self.GAP = GlobalAveragePooling(kernel_size=(afterpq_h//2,afterpq_w//2),stride=1)
#         self.channel_mapper = ConvModule(**channel_mapper)
   
#     def init_weights(self) -> None:
#         return super().init_weights()

    
#     def forward(self,x):    # x ([128,1024,16,8],)
#         mapper_out = self.channel_mapper(x[0])  # mapper_out[128,256,16,8]
#         backbone_out = self.PartQuerier([mapper_out]) # backbone_out {hidden_state [layer=4,dim=128,num_queries=256,hw=256]}
#         hidden_state = backbone_out['hidden_states'][-1] # hidden_state : the last layer output:[dim=128,num_queries=256,hw=256]
        
#         assert self.h * self.w == hidden_state.shape[-1] ,\
#         "Error in projection from 1d to 2d [agg_h:{} agg_w:{}]".format(self.h,self.w)
#         hidden_state = hidden_state.view(-1,self.num_queries,self.h,self.w)     # hidden_state [128,256,16,16]
        
#         neck_out = self.PartAggregator([hidden_state])      # neck_out [1,2048,8,8]
#         out = self.GAP(neck_out)
#         return (out,)
        
    
