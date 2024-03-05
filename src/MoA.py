#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:06:29 2024

@author: umbertocappellazzo
"""


from src.AST_adapters import Bottleneck_adapter, Convpass_adapter, ASTOutput_adapter

import torch 
import torch.nn as nn
from transformers import ASTModel
from dataclasses import dataclass
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer, ASTEncoder
import torch.nn.functional as F
from typing import Optional, Tuple, Union



# For now the MoA can be placed in parallel only. Future implementation may include the sequential version as well.

def make_adapter(in_dim, reduction_rate, out_dim, adapter_block):
    if adapter_block == 'bottleneck' :
        adapter_layer = Bottleneck_adapter(in_dim, reduction_rate, out_dim)
        return adapter_layer
    elif adapter_block == 'convpass':
        adapter_layer = Convpass_adapter(in_dim, reduction_rate, out_dim)
        return adapter_layer
    else:
        raise Exception('Only convpass and bottleneck adapters are supported as of now!')


class Router(nn.Module):
    def __init__(self, model_dim, num_adapters):
        super().__init__()
        
        self.model_dim, self.num_adapters = model_dim, num_adapters
        self.ff = nn.Linear(self.model_dim, self.num_adapters)
        
    def forward(self,x):
        # x shape: [B,L,H]  B = batch_size, L = sequence_length, H = hidden_size
        
        adapter_logits = self.ff(x)  # adapter_logits: [B,L,N], N = number_of_adapters
        adapter_probs = F.softmax(adapter_logits, dim=-1)   
        
        return adapter_probs     #,adapter_logits

class MoA(nn.Module):
    def __init__(self, moa_config, in_dim, reduction_rate):
        super().__init__()
        
        self.moa_config = moa_config
        self.moa = nn.ModuleList([make_adapter(in_dim, reduction_rate, in_dim, moa_config.ADAPTER_MODULE) for _ in range(moa_config.NUM_ADAPTERS)])
    
    def forward(self, x):
        # x shape: [B,L,H]  B = batch_size, L = sequence_length, H = hidden_size
        
        output = torch.stack([adapter_module(x) for adapter_module in self.moa], dim=-1)
        
        return output


def l2_normalize(x, axis, eps=1e-6):
    norm = torch.sqrt(torch.square(x).sum(axis=axis, keepdims=True))
    return x*torch.reciprocal(norm+eps)

class SoftMoA(nn.Module):
    def __init__(self, moa_config, in_dim, reduction_rate):
        super().__init__()
        
        # We follow the notation of the original paper.
        # b=g: group/batch size
        # m: sequence length/# tokens.
        # n: number of experts.
        # p: number of slots per expert.
        # n*p: number of total slots.
        
        self.d = in_dim
        self.n = moa_config.NUM_ADAPTERS
        self.p = moa_config.NUM_SLOTS
        
        self.moa = nn.ModuleList([make_adapter(self.d, reduction_rate, self.d, moa_config.ADAPTER_MODULE) for _ in range(self.n)])
        
        # We use xavier normal initialization (aka Glorot init.). for the slot embeddings.
        self.slot_params = nn.Parameter(torch.empty(self.d, self.n, self.p))
        nn.init.xavier_normal_(self.slot_params)
        
        self.normalize = moa_config.NORMALIZE
        if self.normalize:
            self.scale = nn.Parameter(torch.ones([]))
        
        # We can add a class attribute C if we want to have access to the combine weights for the ablation studies.
        #self.C = None
            
    def forward(self, X):
        # x shape: [b,m,d]. self.slot_params shape: [d,n,p].
        b, m, d = X.shape
        
        if self.normalize:
            X = l2_normalize(X, axis= -1)
            slot_params = l2_normalize(self.slot_params, axis=1)*self.scale
            
        logits = torch.einsum('bmd,dnp->bmnp', X, slot_params) if self.normalize else torch.einsum('bmd,dnp->bmnp', X, self.slot_params) # Shape: [bmnp].
        
        # Compute the dispatch and combine weights. 
        D = torch.softmax(logits, dim= 1) # "Dispatch weights" matrix. 
        # Curiously, jax softmax function allows to pass multiple axes, whereas pytorch implem only one. Reshape here is necessary.
        C = torch.softmax(logits.reshape(b,m,-1), dim=2).reshape(b,m,self.n,self.p) # "Combine weights" matrix.
        
        #self.C = C
        # The input slots are a weighted average of all the input tokens, given by the dispatch weights.
        
        X_tilde = torch.einsum('bmd,bmnp->bnpd', X, D) # Shape: [bnpd].
        
        #Apply the corresponding expert function to each input slot.
        Y_tilde = torch.stack([adapter_module(X_tilde[:,i,:,:]) for i, adapter_module in enumerate(self.moa)], dim= 1)
        
        #The output tokens are a weighted average of all the output slots, given by the combine weights.
        Y = torch.einsum('bnpd,bmnp->bmd', Y_tilde, C)  # Shape: [bmd].
        
        return Y

@dataclass
class SoftMoA_config:
    NUM_ADAPTERS: int 
    NUM_SLOTS: int
    NORMALIZE: bool # Whether to normalize the input and slot_params (aka Phi). As suggested in the paper, the normalization 
                    # has an impact when the hidden_size is increased. In our case, for d=768, there's no difference.
    
    REDUCTION_RATE: int 
    ADAPTER_TYPE: str  # Pfeiffer/Houlsby
    ADAPTER_LOCATION: str  # MHSA/FFN.
    ADAPTER_MODULE: str   # convpass/bottleneck. 


class ASTModel_SoftMoA(ASTModel):
    def __init__(self, config, moa_config: SoftMoA_config):
        super().__init__(config)
        
        self.moa_config= moa_config
        
        self.encoder = ASTEncoder_SoftMoA(config, moa_config)
        
class ASTEncoder_SoftMoA(ASTEncoder):
    def __init__(self, config, moa_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_SoftMoA(config, moa_config) for _ in range(config.num_hidden_layers)])

class ASTLayer_SoftMoA(ASTLayer):
    def __init__(self, config, moa_config):
        super().__init__(config)
        
        self.moa_config = moa_config
        
        # This adapter class adds the residual outside the ASTOutput layer.
        self.output = ASTOutput_adapter(config)
        
        if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
            self.moa = SoftMoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
        else:
            self.moa_mhsa = SoftMoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
            self.moa_ffn = SoftMoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        
        
        output_LN1 = self.layernorm_before(hidden_states)
            
        self_attention_outputs = self.attention(
            output_LN1,  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            
        if self.moa_config.ADAPTER_TYPE == 'Houlsby' or self.moa_config.ADAPTER_LOCATION == 'MHSA':
            if self.moa_config.ADAPTER_TYPE == 'Houlsby':
                moa_output = self.moa_mhsa(output_LN1) 
            else:
                moa_output = self.moa(output_LN1)  
            
            hidden_states = moa_output + attention_output + hidden_states
            
        else:
            hidden_states = attention_output + hidden_states
        
        output_LN2 = self.layernorm_after(hidden_states)
        
        output_up_proj = self.intermediate(output_LN2)
        output_down_proj = self.output(output_up_proj)
        
        if self.moa_config.ADAPTER_TYPE == 'Houlsby' or self.moa_config.ADAPTER_LOCATION == 'FFN':
            
            if self.moa_config.ADAPTER_TYPE == 'Houlsby':
                moa_output = self.moa_ffn(output_LN2)  # Shape: [B,L,H,N].
            else:
                moa_output = self.moa(output_LN2)  # Shape: [B,L,H,N].
            
        hidden_states = hidden_states + output_down_proj + moa_output
        
        outputs = (hidden_states,) + outputs

        return outputs


class AST_SoftMoA(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: tuple, adapter_type: str, location: str, adapter_module: list, num_adapters: int, num_slots: int, normalize: bool, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        assert location in ['MHSA','FFN'], ("Only MHSA and FFN are accepted!")
        assert final_output in ['CLS', 'ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        assert adapter_type in ['Pfeiffer', 'Houlsby', ('Only Pfeiffer and Houlsby are supported!')]
        
        self.moa_config = SoftMoA_config(num_adapters, num_slots, normalize,reduction_rate, adapter_type, location, adapter_module)
        self.model = ASTModel_SoftMoA.from_pretrained(model_ckpt, self.moa_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_moa()
        
    def _unfreeze_moa(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
                self.encoder.layer[block_idx].moa.requires_grad_(True)
                if self.moa_config.ADAPTER_LOCATION == 'MHSA':
                    self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
                else:
                    self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            else:
                self.encoder.layer[block_idx].moa_ffn.requires_grad_(True)
                self.encoder.layer[block_idx].moa_mhsa.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
                
    
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
                    self.encoder.layer[block_idx].moa.train()
                else:
                    self.encoder.layer[block_idx].moa_ffn.train()
                    self.encoder.layer[block_idx].moa_mhsa.train()
            
            self.layernorm.train() 
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    
    def forward(self, x):
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)

        if self.final_output == 'CLS':
            return self.classification_head(hidden_states[:,0])
        else:
            return self.classification_head(hidden_states.mean(dim=1))


      

@dataclass
class MoA_config:
    # MoA details.
    
    NUM_ADAPTERS: int
    REDUCTION_RATE: int
    ADAPTER_TYPE: str  # Pfeiffer/Houlsby.
    ADAPTER_LOCATION: str  # MHSA/FFN. If ADAPTER_TYPE == Houlsby, this parameter is not considered.
    ADAPTER_MODULE: str   # Convpass/Bottleneck.

class ASTModel_MoA(ASTModel):
    def __init__(self, config, moa_config: MoA_config):
        super().__init__(config)
        
        self.moa_config= moa_config
        self.encoder = ASTEncoder_MoA(config, moa_config)
        
class ASTEncoder_MoA(ASTEncoder):
    def __init__(self, config, moa_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_MoA(config, moa_config) for _ in range(config.num_hidden_layers)])

class ASTLayer_MoA(ASTLayer):
    def __init__(self, config, moa_config):
        super().__init__(config)
        
        self.moa_config = moa_config
        
        # This adapter class adds the residual outside the ASTOutput layer.
        self.output = ASTOutput_adapter(config)
        
        if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
            self.router = Router(config.hidden_size, self.moa_config.NUM_ADAPTERS)
            self.moa = MoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
        else:
            self.moa_mhsa = MoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
            self.moa_ffn = MoA(self.moa_config, config.hidden_size, self.moa_config.REDUCTION_RATE)
            self.router_mhsa = Router(config.hidden_size, self.moa_config.NUM_ADAPTERS)
            self.router_ffn = Router(config.hidden_size, self.moa_config.NUM_ADAPTERS)
            
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        output_LN1 = self.layernorm_before(hidden_states)
            
        self_attention_outputs = self.attention(
            output_LN1,  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            
        if self.moa_config.ADAPTER_TYPE == 'Houlsby' or self.moa_config.ADAPTER_LOCATION == 'MHSA':
            if self.moa_config.ADAPTER_TYPE == 'Houlsby':
                router_probs = self.router_mhsa(output_LN1) # Shape: [B,L,N].
                moa_output = self.moa_mhsa(output_LN1)  # Shape: [B,L,H,N].
            else:
                router_probs = self.router(output_LN1) # Shape: [B,L,N].
                moa_output = self.moa(output_LN1)  # Shape: [B,L,H,N].
            
            moa_output = (router_probs[:,:,None,:]*moa_output).sum(-1)
            
            hidden_states = moa_output + attention_output + hidden_states
            
        else:
            hidden_states = attention_output + hidden_states
        
        output_LN2 = self.layernorm_after(hidden_states)
        
        output_up_proj = self.intermediate(output_LN2)
        output_down_proj = self.output(output_up_proj)
        
        if self.moa_config.ADAPTER_TYPE == 'Houlsby' or self.moa_config.ADAPTER_LOCATION == 'FFN':
            
            if self.moa_config.ADAPTER_TYPE == 'Houlsby':
                router_probs = self.router_ffn(output_LN2) # Shape: [B,L,N].
                moa_output = self.moa_ffn(output_LN2)  # Shape: [B,L,H,N].
            else:
                router_probs = self.router(output_LN2) # Shape: [B,L,N].
                moa_output = self.moa(output_LN2)  # Shape: [B,L,H,N].
            
            moa_output = (router_probs[:,:,None,:]*moa_output).sum(-1) # Shape: [B,L,H].
        
        hidden_states = hidden_states + output_down_proj + moa_output
        
        outputs = (hidden_states,) + outputs

        return outputs


class AST_MoA(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, location: str, adapter_module: str, num_adapters: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        assert location in ['MHSA','FFN'], ("Only MHSA and FFN are accepted!")
        assert final_output in ['CLS', 'ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        assert adapter_type in ['Pfeiffer', 'Houlsby', ('Only Pfeiffer and Houlsby are supported!')]
        
        self.moa_config = MoA_config(num_adapters, reduction_rate, adapter_type, location, adapter_module)
        self.model = ASTModel_MoA.from_pretrained(model_ckpt, self.moa_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_moa()
        
    def _unfreeze_moa(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
                self.encoder.layer[block_idx].moa.requires_grad_(True)
                self.encoder.layer[block_idx].router.requires_grad_(True)
                if self.moa_config.ADAPTER_LOCATION == 'MHSA':
                    self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
                else:
                    self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            else:
                self.encoder.layer[block_idx].moa_ffn.requires_grad_(True)
                self.encoder.layer[block_idx].moa_mhsa.requires_grad_(True)
                self.encoder.layer[block_idx].router_mhsa.requires_grad_(True)
                self.encoder.layer[block_idx].router_ffn.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
                
    
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                if self.moa_config.ADAPTER_TYPE == 'Pfeiffer':
                    self.encoder.layer[block_idx].router.train()
                    self.encoder.layer[block_idx].moa.train()
                else:
                    self.encoder.layer[block_idx].router_ffn.train()
                    self.encoder.layer[block_idx].moa_ffn.train()
                    self.encoder.layer[block_idx].router_mhsa.train()
                    self.encoder.layer[block_idx].moa_mhsa.train()
            
            self.layernorm.train() 
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    
    def forward(self, x):
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)

        if self.final_output == 'CLS':
            return self.classification_head(hidden_states[:,0])
        else:
            return self.classification_head(hidden_states.mean(dim=1))