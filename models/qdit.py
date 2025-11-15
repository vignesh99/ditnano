#general libraries
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Union, Dict
#from models.models import TimestepEmbedder, LabelEmbedder, DiTBlock

#custom libraries
from typing import Tuple
from src.module.base import _QBaseLinear, _QBaseConv2d, _QBase
from src.module.attention import QAttention, QWindowAttention, QBertSelfAttention
from src.quantization.adaround import AdaRound
from src.quantization.lsq import LSQ, LSQTokenWise
from src.quantization.qdrop import QDrop, QDropTokenWise
from src.quantization.minmax import MinMaxQuantizer, MinMaxTokenWiseQuantizer, MinMaxChannelWiseWeightQuantizer, MinMaxChannelWiseActQuantizer
from src.quantization.observer import BaseObserver, BaseChannelWiseObserver, BaseTokenWiseObserver
from src.quantization.smoothquant import SmoothQuantizer, SmoothQuantChannelWiseWeightQuantizer, SmoothQuantTokenWiseQuantizer

#quantization type dictionaries
weight_quantizer = {
    "adaround": AdaRound,
    "minmax": MinMaxQuantizer,
    "minmax_channel": MinMaxChannelWiseWeightQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_channel": SmoothQuantChannelWiseWeightQuantizer,
    "identity": _QBase
}

input_quantizer = {
    "minmax": MinMaxQuantizer,
    "minmax_token": MinMaxTokenWiseQuantizer,
    "minmax_channel": MinMaxChannelWiseActQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_token": SmoothQuantTokenWiseQuantizer,
    "lsq": LSQ,
    "lsq_token": LSQTokenWise,
    "qdrop": QDrop,
    "qdrop_token": QDropTokenWise,
    "identity": _QBase
}


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class QuantizedTimestepEmbedder(QuantizedModel) :
    def __init__(self, org_model, **quant_params) :
        #set up the quantization params
        self.quant_dict = quant_params['quant_dict']
        super().__init__()
        quant_params_ = quant_params.copy()
        if 'Et' in self.quant_dict:
            quant_params_['weight_range_method'] = RangeEstimators.MSE
            quant_params_['weight_range_options'] = dict(opt_method=OptMethod.golden_section)
        #quantize the FP mlp
        self.mlp = quantize_model(org_model.mlp, **quant_params_)
        self.frequency_embedding_size = org_model.frequency_embedding_size
        #quantized activation for forward pass
        self.tfreq_quantizer = QuantizedActivation(**quant_params)
        self.temb_quantizer = QuantizedActivation(**quant_params)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = self.tfreq_quantizer(t_freq)
        t_emb = self.mlp(t_freq)
        t_emb = self.temb_quantizer(t_emb)
        return t_emb
