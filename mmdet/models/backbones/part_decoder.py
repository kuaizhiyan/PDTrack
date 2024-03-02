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
from mmdet.models.necks import ChannelMapper

@MODELS.register_module()
class PartQuerier(BaseModule):
    """ Learn part features by part querier from feature maps.
    
    Args:
        num_queries: int the number of part query
    
    """
    
    def __init__(self,
                 embed_dims:int = 256,
                 decoder: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 16,
                 with_agg:bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 ) -> None:
        super(PartQuerier,self).__init__()
        # process args
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.decoder = decoder      # cfg
        self.positional_encoding = positional_encoding  # cfg
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.with_agg = with_agg

        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        # self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = ConditionalDetrTransformerDecoder(**self.decoder)
        # self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'   
    
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
        # for coder in self.encoder, self.decoder:
        #     for p in coder.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)
    
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
        ) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """
        if not self.with_agg:
            feat = img_feats[-1]  # NOTE img_feats contains only one feature.   # [4,3,32,16]
            batch_size, feat_dim, _, _ = feat.shape
            
            # construct binary masks which for the transformer.    
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat) # feat[1,2048,8,4] pos_emb [1,256,8,4]
        
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            # [bs, c, h, w] -> [bs, h*w, c]
            feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
            # [bs, h, w] -> [bs, h*w]
            if masks is not None:
                masks = masks.view(batch_size, -1)

            # prepare transformer_inputs_dict
            encoder_inputs_dict = dict(
                feat=feat, feat_mask=masks, feat_pos=pos_embed)
            decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
            return encoder_inputs_dict, decoder_inputs_dict 
        else:
            all_feat = None
            all_pos_embed = None
            for i,feat in enumerate(img_feats):
                feat = img_feats[i]  # NOTE img_feats contains only one feature.   # [4,3,32,16]
                batch_size, feat_dim, _, _ = feat.shape
                
                # construct binary masks which for the transformer.    
                masks = None
                # [batch_size, embed_dim, h, w]
                pos_embed = self.positional_encoding(masks, input=feat) # feat[1,2048,8,4] pos_emb [1,256,8,4]
            
                # use `view` instead of `flatten` for dynamically exporting to ONNX
                # [bs, c, h, w] -> [bs, h*w, c]
                feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
                pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
                # [bs, h, w] -> [bs, h*w]
                # if masks is not None:
                #     masks = masks.view(batch_size, -1)
                if all_feat is None:
                    all_feat = feat
                else:
                    all_feat = torch.cat([all_feat,feat],dim=1)
                if all_pos_embed is None :
                    all_pos_embed = pos_embed
                else:
                    all_pos_embed = torch.cat([all_pos_embed,pos_embed],dim=1)

            # prepare transformer_inputs_dict
            encoder_inputs_dict = dict(
                feat=all_feat, feat_mask=masks, feat_pos=all_pos_embed)
            decoder_inputs_dict = dict(memory_mask=masks, memory_pos=all_pos_embed)
            return encoder_inputs_dict, decoder_inputs_dict
        
    
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = self.query_embedding.weight
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references` of the decoder output.

            - hidden_states (Tensor): Has shape
                (num_decoder_layers, bs, num_queries, dim)
            - references (Tensor): Has shape
                (bs, num_queries, 2)
        """

        hidden_states, references = self.decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references)
        return head_inputs_dict
    
    def forward(self,img_feats):    
        
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats)  # img_feats tuple([1,256,25,38]) samples[..,img_shape=800,1199]
        # encoder_inputs:{feat:[1,950,256],feat_mask:None,feat_pos[1,950,256]} decoder:{'mem_pos'[1,950,256]}
         
        # 不通过 ENcoder 了
        # encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict) {'memory'[1,950,256]}
        encoder_inputs_dict = {'memory':encoder_inputs_dict['feat']}
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_inputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)  # {query_pos[1,300,256],query[1,300,256],memory[1,950,256]}
        # decoder_inputs_dict {query,query_pos,memory,memory_pos}
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)  #{hidden_state[6,1,300,256],reference[1,300,2]}
        head_inputs_dict.update(decoder_outputs_dict)
        
        return head_inputs_dict #dict{hidden_state[6,1,300,256],reference[1,300,2]}
        
    
@MODELS.register_module()
class PartQuerier_neck(BaseModule):
    """
    The PartQuerier serves as the neck of the ReID network.
    The data processing flow is as :
         +-----------+
         |PartQuerier|
         +-----------+
               |
               V
        +--------------+
        |PartAggregator|
        +--------------+
               |
               V
       +----------------+ 
       |GlobalAvgPooling|
       +----------------+ 
    
    Return:
        out: tuple of outputs
    
    """
    
    def __init__(self, 
                 embed_dims:int=256,
                 with_agg:bool=False,
                 decoder: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 16,
                 channel_mapper:OptConfigType=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None
                 )->tuple:
        super(PartQuerier_neck,self).__init__(init_cfg=init_cfg)
        self.num_queries = num_queries
        self.PartQuerier = PartQuerier(    
                                       embed_dims=embed_dims,
                                       decoder=decoder,
                                       with_agg=with_agg,
                                       positional_encoding=positional_encoding,
                                       num_queries=num_queries,
                                       train_cfg=train_cfg,
                                       test_cfg=test_cfg)
        self.channel_mapper = ChannelMapper(**channel_mapper)
        # self.init_cfg = init_cfg
        
    # def init_weights(self) -> None:
    #     return super().init_weights(self.init_cfg)

    
    def forward(self,x):    # x ([128,1024,16,8],)
        out = self.channel_mapper(x)  # mapper_out:0([128,256,16,8])
        # 这里返回 [num_layers,num_queries,dim]
        out = self.PartQuerier(out) # out{'hidden_states'[4,128,129,256],'references'[1,128,2]}
        # 取最后一层，每个 batch 第一个 token 做分类
        hidden_state = out['hidden_states'][-1][:,0] # hidden_state : the last layer output:[dim=128,num_queries=256,hw=256]
        # hidden_state = hidden_state.unsqueeze(0)
        return (hidden_state,)
        
    
