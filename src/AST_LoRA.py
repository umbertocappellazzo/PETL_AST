#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:53:39 2023

@author: umbertocappellazzo
"""


######################
#                    #   
#   Code for LoRA.   #
#                    #
######################



import torch 
import torch.nn as nn
from transformers import ASTModel
import math
from dataclasses import dataclass
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer, ASTEncoder, ASTAttention, ASTSelfAttention
from typing import Optional, Tuple, Union


# LoRA implementation for AST model.

@dataclass
class LoRA_config:
    RANK: int
    ALPHA: int = 1
    

class ASTModel_LoRA(ASTModel):
    def __init__(self, config, lora_config: LoRA_config):
        super().__init__(config)
        
        self.lora_config = LoRA_config
        
        self.encoder = ASTEncoder_LoRA(config, lora_config)
    
    
class ASTEncoder_LoRA(ASTEncoder):
    def __init__(self, config, lora_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_LoRA(config, lora_config) for _ in range(config.num_hidden_layers)])     


class ASTLayer_LoRA(ASTLayer):
    def __init__(self, config, lora_config):
        super().__init__(config)  
    
        self.attention = ASTAttention_LoRA(config, lora_config)


class ASTAttention_LoRA(ASTAttention):
    def __init__(self, config, lora_config):
        super().__init__(config)
        
        self.attention = ASTSelfAttention_LoRA(config, lora_config)

class ASTSelfAttention_LoRA(ASTSelfAttention):
    def __init__(self, config, lora_config):
        super().__init__(config)
    
        self.rank = lora_config.RANK
        self.scaling = lora_config.ALPHA/self.rank
        
        hid_size = config.hidden_size
        
        self.lora_down = nn.Linear(hid_size, round(hid_size/self.rank), bias= False)
        self.lora_up = nn.Linear(round(hid_size/self.rank), hid_size, bias= False)
        
        # As in the original paper, matrix A (lora_down) and B (lora_up) use 
        # zero and Gaussian initialization, respectively. 
        nn.init.zeros_(self.lora_down.weight)
        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
    
    
    # LoRA is applied to QUERY and KEY projection matrices, not VALUE!!
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]: 
        
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        layer_lora = self.lora_up(self.lora_down(hidden_states))
        query_layer = query_layer + layer_lora*self.scaling
        key_layer = key_layer + layer_lora*self.scaling
        
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # From this point forward, the code is the same as ASTSelfAttention layer.
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
        


class AST_LoRA(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, rank: int, alpha: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.lora_config = LoRA_config(rank, alpha)
        self.model = ASTModel_LoRA.from_pretrained(model_ckpt, self.lora_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_loras()
        
    def _unfreeze_loras(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].attention.attention.lora_down.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_up.requires_grad_(True)
            
            # Optional: finetune also the LayerNorm before the MHSA layer.
            #self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
            
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            
            # Just in case the LayerNorm before MHSA is finetuned.
            #for block_idx in range(self.model_config.num_hidden_layers):
            #    self.encoder.layer[block_idx].layernorm_before.train()
            
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