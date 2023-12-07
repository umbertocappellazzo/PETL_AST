#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:55:19 2023

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from transformers import ASTModel
import math
from operator import mul
from functools import reduce
from torch.nn import Dropout
from torch.nn.modules.utils import _pair
from dataclasses import dataclass
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer, ASTEncoder, ASTAttention, ASTSelfAttention
from typing import Optional, Tuple, Union


#Code for prompt-tuning (Shallow and Deep) and prefix-tuning.
#The code for prompt-tuning is adapted from https://github.com/kmnp/vpt


# PROMPT configuration class.

@dataclass
class Prompt_config:
    NUM_TOKENS: int = 5
    LOCATION: str = "prepend"
    INITIATION: str = "random"
    DEEP: bool = False
    DROPOUT: float = 0.0
    FINAL_OUTPUT: str = "CLS"


# AST class for Prompt-Tuning training.

class PromptAST(nn.Module):

    def __init__(self, prompt_config: Prompt_config, max_length: int, num_classes: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):

        super().__init__()
        
        self.prompt_config = prompt_config
        
        assert (self.prompt_config.FINAL_OUTPUT in ['CLS','ALL','PROMPTS','SPEECH','PROMPTS+SPEECH']), ('FINAL_OUTPUT parameter not included in the supported configs!')

        base_model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = base_model.config
        self.final_output = self.prompt_config.FINAL_OUTPUT
        
        self.embeddings = base_model.embeddings
        self.encoder = base_model.encoder
        self.layernorm = base_model.layernorm

        self.patch_size = _pair(self.model_config.patch_size)
        
        # Prompt Config
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # Prompt PROJECT (No projection for now)
        prompt_dim = self.model_config.hidden_size
        
        self.prompt_proj = nn.Identity()
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)

        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)

        # INITIATE PROMPT
        if self.prompt_config.INITIATION == "random":

            val =  math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + prompt_dim)) # patch_size taken from AST CONFIG
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim
            ))

            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, +val)

            # DEEP PROMPT INITIALIZATION
            if self.prompt_config.DEEP:
                total_d_layer = self.model_config.num_hidden_layers-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
    
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.layernorm.train()  #self.layernorm.eval()      
            self.prompt_proj.train()
            self.prompt_dropout.train()
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def incorporate_prompt(self, x):
        B = x.shape[0]
  
        x = self.embeddings(x)  # (batch_size, 2 + n_patches, hidden_dim)
        
        # The distillation_token at position 1 is discarded as it is not useful for us.
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 2:, :]
        ), dim=1)

        return x
    
    
    # DEEP PROMPT FORWARDING
    def forward_deep_prompt(self, embedding_output):
        hidden_states = None
        B = embedding_output.shape[0]
        num_layers = self.model_config.num_hidden_layers
        
        for i in range(num_layers):
            if i == 0:
                
                hidden_states = self.encoder.layer[i](embedding_output)[0]
                
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                    
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)
 
                hidden_states = self.encoder.layer[i](hidden_states)[0]
                
               
        return hidden_states
    
    def forward_tsne(self,x):
        embedding_output = self.incorporate_prompt(x)
        if self.prompt_config.DEEP:
            hidden_states = self.forward_deep_prompt(
                 embedding_output)
            hidden_states = self.layernorm(hidden_states)
        
            return hidden_states[:,0], hidden_states.mean(dim=1)
        else: 
            hidden_states = self.encoder(embedding_output)[0]
            hidden_states = self.layernorm(hidden_states)
            return hidden_states[:,0], hidden_states.mean(dim=1)
            
    def forward(self, x):
        
        embedding_output = self.incorporate_prompt(x)
       
        if self.prompt_config.DEEP:
            
            hidden_states = self.forward_deep_prompt(
                 embedding_output)
            hidden_states = self.layernorm(hidden_states)
            
            if self.final_output == 'CLS':
                out = self.classification_head(hidden_states[:,0])
            elif self.final_output == 'ALL':
                out = self.classification_head(hidden_states.mean(dim=1))
            elif self.final_output == 'PROMPTS':
                 out = self.classification_head(hidden_states[:,1:(self.num_tokens+1)].mean(dim=1))
            elif self.final_output == 'SPEECH':
                 out = self.classification_head(hidden_states[:,1+self.num_tokens:].mean(dim=1))
            elif self.final_output == 'PROMPTS+SPEECH':
                out = self.classification_head(hidden_states[:,1:].mean(dim=1))
                
        else:
            hidden_states = self.encoder(embedding_output)[0]
            hidden_states = self.layernorm(hidden_states)
            
            if self.final_output == 'CLS':
                out = self.classification_head(hidden_states[:,0])
            elif self.final_output == 'ALL':
                out = self.classification_head(hidden_states.mean(dim=1))
            elif self.final_output == 'PROMPTS':
                 out = self.classification_head(hidden_states[:,1:(self.num_tokens+1)].mean(dim=1))
            elif self.final_output == 'SPEECH':
                 out = self.classification_head(hidden_states[:,1+self.num_tokens:].mean(dim=1))
            elif self.final_output == 'PROMPTS+SPEECH':
                out = self.classification_head(hidden_states[:,1:].mean(dim=1))
        
        return out


# Prefix Tuning implementation for AST model.

@dataclass
class Prefix_tuning_config:
    # PATCH_SIZE and PROMPT_DIM are used to initialize the prompts (i.e., xavier_uniform initialization).
    
    NUM_TOKENS: int
    PATCH_SIZE: int
    PROMPT_DIM: int

# pt = Prefix Tuning

class ASTModel_PT(ASTModel):
    def __init__(self, config, pt_config: Prefix_tuning_config):
        super().__init__(config)
        
        self.pt_config = pt_config
        self.encoder = ASTEncoder_PT(config, pt_config)
    
    
class ASTEncoder_PT(ASTEncoder):
    def __init__(self, config, pt_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_PT(config, pt_config) for _ in range(config.num_hidden_layers)])     


class ASTLayer_PT(ASTLayer):
    def __init__(self, config, pt_config):
        super().__init__(config)  
    
        self.attention = ASTAttention_PT(config, pt_config)


class ASTAttention_PT(ASTAttention):
    def __init__(self, config, pt_config):
        super().__init__(config)
        
        self.attention = ASTSelfAttention_PT(config, pt_config)

class ASTSelfAttention_PT(ASTSelfAttention):
    def __init__(self, config, pt_config):
        super().__init__(config)
    
        self.num_tokens = pt_config.NUM_TOKENS
        self.prompt_dim = pt_config.PROMPT_DIM
        
        patch_size = _pair(pt_config.PATCH_SIZE)
        val =  math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.prompt_dim))
        
        # The num_tokens tokens are splitted equeally and appended to the key and value vectors.
        
        self.key_tokens = nn.Parameter(torch.zeros(
            1, int(self.num_tokens/2), self.prompt_dim
        ))
        self.value_tokens = nn.Parameter(torch.zeros(
            1, int(self.num_tokens/2), self.prompt_dim
        ))
        # xavier_uniform initialization
        nn.init.uniform_(self.key_tokens, -val, +val)
        nn.init.uniform_(self.value_tokens, -val, +val)
    
    
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]: 
        
        B = hidden_states.shape[0]
        
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        key_layer = torch.cat((
                        key_layer[:, :1, :],
                        self.key_tokens.expand(B, -1, -1),
                        key_layer[:, 1:, :]
            ), dim=1)
        value_layer = torch.cat((
                        value_layer[:, :1, :],
                        self.value_tokens.expand(B, -1, -1),
                        value_layer[:, 1:, :]
            ), dim=1)
        

        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        query_layer = self.transpose_for_scores(query_layer)
        
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


class AST_Prefix_tuning(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, num_tokens: int, patch_size: int, hidden_size: int,  model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        assert (num_tokens % 2) == 0, ("The number of prompts must be even as it has to be splitted equally across the key and value vectors!")
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.pt_config = Prefix_tuning_config(num_tokens, patch_size, hidden_size)
        self.model = ASTModel_PT.from_pretrained(model_ckpt, self.pt_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_prompts()
        
    def _unfreeze_prompts(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].attention.attention.key_tokens.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.value_tokens.requires_grad_(True)
            
            # Optional: finetune also the LayerNorm befor the MHSA layer.
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