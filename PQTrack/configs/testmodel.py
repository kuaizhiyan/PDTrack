__base__ = ["./configs/_base_/datasets/mot_challenge_reid.py",
            "./configs/_base_/default_runtime.py"]


"""
channel_mapper.in_channels = feature map channel
              .out_channels = feat_dim (emb_dim)

encoder/decoder.layer_cfg.embed_dims == feat_dim (from backbone) == 2 x pos_emb.num_feats

aggregator.inplanes = num_queriers, out_planse=2048. 
The aggregator downsample 4 times and must to be (b,2048,8,4)

"""

image_scale = (128,256)
_num_queries=256
model = dict(
    type='TestModel_neck',
        num_queries=256,
        with_encoder=False,
        with_decoder=True,
        with_conditionpos=False,
        with_agg=True,
        channel_mapper=dict(
            in_channels=[512,1024,2048],   # the output feature map dim
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='LeakyReLU')
            ),
        encoder=dict(
            num_layers=4,
            layer_cfg=dict(  # DetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)))
        ),
        decoder=dict(
            num_layers=4,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=False),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=True))
        ),
        positional_encoding=dict(num_feats=128, normalize=True),    # num_feats = len(x)+len(y)

)