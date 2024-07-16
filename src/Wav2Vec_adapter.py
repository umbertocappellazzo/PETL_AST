#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:46:32 2023

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from transformers import Wav2Vec2Model 
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, Wav2Vec2EncoderLayer
import math
from operator import mul
from functools import reduce
from torch.nn import Dropout
from torch.nn.modules.utils import _pair
from dataclasses import dataclass


# CONFORMER adapter module.

class Conformer_adapter(nn.Module):
    """Implements the conformer convolution module
    as described in https://arxiv.org/abs/2005.08100
    Args:
        d_model (int): The model dimension.
        kernel_size (int): The depth-wise convolution kernel size.
        p_dropout (float): The dropout rate.
    """

    def __init__(self, in_dim: int, out_dim,  kernel_size: int, p_dropout: float, reduction_rate) -> None:
        super().__init__()
        bottleneck_dim = round(in_dim/reduction_rate)
        
        self.lnorm = nn.LayerNorm(normalized_shape=in_dim)
        self.pwise_conv1 = nn.Conv1d(
            in_channels=in_dim, out_channels=bottleneck_dim*2, kernel_size=1
        )
        self.act1 = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            kernel_size=kernel_size,
            groups=bottleneck_dim,
            padding="same",
            #dilation=3
        )
        self.bnorm = nn.BatchNorm1d(num_features=bottleneck_dim)
        self.act2 = nn.SiLU()
        self.pwise_conv2 = nn.Conv1d(
            in_channels=bottleneck_dim, out_channels=out_dim, kernel_size=1
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """
        Passes the input tensor through the Conformer Convolutional Module.
        Args:
            x (Tensor): Input tensor of shape [B, M, d].
        Returns:
            Tensor: Result tensor of shape [B, M, d].
        """

        out = self.lnorm(x)
        out = out.transpose(-1, -2)  # [B, d, M]
        out = self.pwise_conv1(out)  # [B, 2d, M]
        out = self.act1(out)  # [B, d, M]
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.act2(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        out = out.transpose(-1, -2)  # [B, M, d]
        return out
        


# CONVPASS adapter module.

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Convpass_adapter(nn.Module):
    def __init__(self, in_dim, reduction_rate, out_dim):
        super().__init__()
        
        bottleneck_dim = round(in_dim/reduction_rate)
        self.adapter_conv = nn.Conv1d(bottleneck_dim , bottleneck_dim, 3, padding="same")
        
        nn.init.xavier_uniform_(self.adapter_conv.weight)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(in_dim, bottleneck_dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(bottleneck_dim, out_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x):
      
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_conv = self.adapter_conv(x_down.transpose(1,2))
        x_conv = x_conv.transpose(1,2)

        x_down = self.dropout(self.act(x_down))
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


# BOTTLENECK adapter module.

class Bottleneck_adapter(nn.Module):
    def __init__(self, in_dim, reduction_rate, out_dim):
        super().__init__()
        
        bottleneck_dim = round(in_dim/reduction_rate)
        self.linear_downsample = nn.Linear(in_dim, bottleneck_dim)
        self.linear_upsample = nn.Linear(bottleneck_dim, out_dim)
        #self.layer_norm_adapt = nn.LayerNorm(out_dim)
        self.act = torch.nn.GELU()
        
        nn.init.zeros_(self.linear_downsample.weight); nn.init.zeros_(self.linear_upsample.weight)
        nn.init.zeros_(self.linear_downsample.bias); nn.init.zeros_(self.linear_upsample.bias);
        
    def forward(self, x):
        down_x = self.linear_downsample(x)
        up_x = self.linear_upsample(self.act(down_x))
        
        return up_x
        #return self.layer_norm_adapt(up_x)



# Standard Wav2Vec model.


class Wav2Vec(nn.Module):
    def __init__(self, num_classes: int, model_ckpt="facebook/wav2vec2-base-960h"):
        super().__init__()
        
        self.model = Wav2Vec2Model.from_pretrained(model_ckpt)
        self.model.feature_extractor.requires_grad_(False)
        self.model_config = self.model.config
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
    def forward(self, x):
        
        hidden_states = self.model(x)[0]
        
        return self.classification_head(hidden_states.mean(dim=1))



# Adapter-tuning.

@dataclass
class Adapter_config:
    REDUCTION_RATE: int 
    ADAPTER_TYPE: str
    ADAPTER_CONF: str
    APPLY_RESIDUAL: bool
    ADAPTER_BLOCK: str
    KERNEL_SIZE: int # the kernel size for the conformer. 
    

class Wav2Vec2Model_adapter(Wav2Vec2Model):
    def __init__(self, config, adapter_config: Adapter_config):
        super().__init__(config)
        
        self.adapter_config= adapter_config
        
        self.encoder = Wav2Vec2Encoder_adapter(config, adapter_config)


class Wav2Vec2Encoder_adapter(Wav2Vec2Encoder):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer_adapter(config, adapter_config) for _ in range(config.num_hidden_layers)])

class Wav2Vec2EncoderLayer_adapter(Wav2Vec2EncoderLayer):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        self.adapter_config = adapter_config
        self.config = config
        
        self.adapter_module_FFN = self.make_adapter(config.hidden_size,adapter_config.REDUCTION_RATE,config.hidden_size,adapter_config.ADAPTER_BLOCK, adapter_config.KERNEL_SIZE)
        
        if adapter_config.ADAPTER_TYPE == 'Houlsby':
            self.adapter_module_MHSA = self.make_adapter(config.hidden_size,adapter_config.REDUCTION_RATE,config.hidden_size,adapter_config.ADAPTER_BLOCK, adapter_config.KERNEL_SIZE)
        
        
    def make_adapter(self, in_dim, reduction_rate, out_dim, adapter_block, kernel_size):
        if adapter_block == 'conformer':
            adapter_layer = Conformer_adapter(in_dim, out_dim, kernel_size, 0., reduction_rate)
            return adapter_layer
        elif adapter_block == 'bottleneck' :
            adapter_layer = Bottleneck_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        else:
            raise Exception('Only conformer and bottleneck adapters are supported as of now!')
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        
        
        attn_residual = hidden_states
        
        
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        
        if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
            if self.adapter_config.ADAPTER_CONF == 'parallel':
                adapter_output_MHSA = self.adapter_module_MHSA(attn_residual)
                hidden_states = hidden_states + attn_residual + adapter_output_MHSA
            else:
                adapter_output_MHSA = self.adapter_module_MHSA(hidden_states)
                if self.adapter_config.APPLY_RESIDUAL:
                    hidden_states = hidden_states + attn_residual + adapter_output_MHSA
                else:
                    hidden_states = attn_residual + adapter_output_MHSA
        else:
            hidden_states = attn_residual + hidden_states
        

        hidden_states = self.layer_norm(hidden_states)
        
        
        if self.adapter_config.ADAPTER_CONF == 'parallel': 
            adapter_output_FFN = self.adapter_module_FFN(hidden_states)
            hidden_states = hidden_states + self.feed_forward(hidden_states) + adapter_output_FFN
        else:
            ffn_output = self.feed_forward(hidden_states)
            if self.adapter_config.APPLY_RESIDUAL:
                hidden_states = hidden_states + ffn_output + self.adapter_module_FFN(ffn_output)
            else:
                hidden_states = hidden_states + self.adapter_module_FFN(ffn_output)
            
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

# Wav2vec class for adapter training.

class Wav2Vec_adapter(nn.Module):
    def __init__(self, num_classes: int, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, kernel_size: int, finetune_LN= False, model_ckpt='facebook/wav2vec2-base-960h'):
        super().__init__()
        
        self.adapter_config = Adapter_config(reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block, kernel_size)
        self.model = Wav2Vec2Model_adapter.from_pretrained(model_ckpt, self.adapter_config)
        
        self.model_config = self.model.config
        self.finetune_LN = finetune_LN
        
        assert adapter_type in ['Pfeiffer','Houlsby'], ('Only Pfeiffer and Houlsby adapter is supported for AST!')
        
        self.feature_extractor = self.model.feature_extractor
        self.feature_projection = self.model.feature_projection
        self.encoder = self.model.encoder
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.feature_extractor.requires_grad_(False)  
        self.feature_projection.requires_grad_(False)
        self.encoder.requires_grad_(False)

        self._unfreeze_adapters()

    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            if self.finetune_LN:
                self.encoder.layers[block_idx].layer_norm.requires_grad_(True)
            self.encoder.layers[block_idx].adapter_module_FFN.requires_grad_(True)
            if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                self.encoder.layers[block_idx].adapter_module_MHSA.requires_grad_(True)
                
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            self.feature_extractor.eval()
            self.feature_projection.eval()
            self.encoder.eval()
            self.classification_head.train()
            
            for block_idx in range(self.model_config.num_hidden_layers):
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layers[block_idx].adapter_module_FFN.bnorm.train()
                    self.encoder.layers[block_idx].adapter_module_FFN.lnorm.train()
            
                if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                    if self.adapter_config.ADAPTER_BLOCK =='conformer':
                        self.encoder.layers[block_idx].adapter_module_MHSA.bnorm.train()
                        self.encoder.layers[block_idx].adapter_module_MHSA.lnorm.train()
                if self.finetune_LN:
                    self.encoder.layers[block_idx].layer_norm.train()
            
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(self, x):
        extract_features = self.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        
        hidden_states, _ = self.feature_projection(extract_features)
        
        hidden_states = self.encoder(hidden_states)[0]
        out = self.classification_head(hidden_states.mean(dim=1))
        
        return out