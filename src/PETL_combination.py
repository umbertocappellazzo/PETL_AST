#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:22:21 2023

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
from src.AST_adapters import Adapter_config, ASTModel_adapter, ASTEncoder_adapter
from src.AST_prompt_tuning import Prompt_config
from src.AST_LoRA import ASTAttention_LoRA


# The following classes combine multiple PETL methods.


# AST class for Prompt-Tuning + adapter training.

class AST_adapterPrompt(nn.Module):
    def __init__(self, prompt_config: Prompt_config, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        ''' The reduction rate decides the bottleneck dimension of the adapter module --> bottleneck_dim = in_dim/reduction_rate.
            The adapter_type param specifies the type of the adapter. Supported types: "Houlsby" and "Pfeiffer".
            LN_train: whether the LN layers are trained along with the adapters. Original papers train the LNs.
        '''
        
        super().__init__()
        
        self.adapter_config = Adapter_config(reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block)
        self.prompt_config = prompt_config
        
        self.model = ASTModel_adapter.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        assert adapter_type in ['Pfeiffer','Houlsby'], ('Only Pfeiffer and Houlsby adapter is supported for AST!')
        
        self.patch_size = _pair(self.model_config.patch_size)
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        prompt_dim = self.model_config.hidden_size
        self.prompt_proj = nn.Identity()
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_adapters()
        
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
        
    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].adapter_module_FFN.requires_grad_(True)
            self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                self.encoder.layer[block_idx].adapter_module_MHSA.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
    
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layer[block_idx].adapter_module_FFN.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module_FFN.lnorm.train()
                
                if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                    self.encoder.layer[block_idx].layernorm_before.train()
                    if self.adapter_config.ADAPTER_BLOCK =='conformer':
                        self.encoder.layer[block_idx].adapter_module_MHSA.bnorm.train()
                        self.encoder.layer[block_idx].adapter_module_MHSA.lnorm.train()
                        
                self.encoder.layer[block_idx].layernorm_after.train()
            
            self.layernorm.train() 
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def incorporate_prompt(self, x):
        B = x.shape[0]
        
        x = self.embeddings(x)
        
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
            # print(i)
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

# LoRA + Adapter implementation for AST model.

@dataclass
class LoRA_Adapter_config:
    RANK: int
    ALPHA: int
    REDUCTION_RATE: int 
    ADAPTER_TYPE: str
    ADAPTER_CONF: str
    APPLY_RESIDUAL: bool
    ADAPTER_BLOCK: str




class ASTModel_LoRA_Adapter(ASTModel):
    def __init__(self, config, lora_adapt_config: LoRA_Adapter_config):
        super().__init__(config)
        
        self.lora_adapt_config = lora_adapt_config
        
        self.encoder = ASTEncoder_LoRA_Adapter(config, lora_adapt_config)

class ASTEncoder_LoRA_Adapter(ASTEncoder_adapter):  
    def __init__(self, config, lora_adapt_config):
        super().__init__(config, lora_adapt_config)
        
        for module in self.layer:
            module.attention = ASTAttention_LoRA(config, lora_adapt_config)


class AST_LoRA_Adapter(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, rank: int, alpha: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        
        self.adapter_config = LoRA_Adapter_config(rank, alpha, reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block)
        self.model = ASTModel_LoRA_Adapter.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        assert adapter_type in ['Pfeiffer','Houlsby'], ('Only Pfeiffer and Houlsby adapter is supported for AST!')
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_loras()
        self._unfreeze_adapters()
        
    def _unfreeze_loras(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].attention.attention.lora_down.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_up.requires_grad_(True)
            
            # Optional: finetune also the LayerNorm befor the MHSA layer.
            #self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)

    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].adapter_module_FFN.requires_grad_(True)
            self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                self.encoder.layer[block_idx].adapter_module_MHSA.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
    
    
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layer[block_idx].adapter_module_FFN.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module_FFN.lnorm.train()
            
                if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                    self.encoder.layer[block_idx].layernorm_before.train()
                    if self.adapter_config.ADAPTER_BLOCK =='conformer':
                        self.encoder.layer[block_idx].adapter_module_MHSA.bnorm.train()
                        self.encoder.layer[block_idx].adapter_module_MHSA.lnorm.train()
                        
                self.encoder.layer[block_idx].layernorm_after.train()
            
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


# LoRA + adapter-tuning + prompt-tuning.

class AST_LoRA_Adapter_Prompt(nn.Module):
    def __init__(self, prompt_config: Prompt_config, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, rank: int, alpha: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.adapter_config = LoRA_Adapter_config(rank, alpha, reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block)
        self.model = ASTModel_LoRA_Adapter.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.prompt_config = prompt_config
        self.final_output = self.prompt_config.FINAL_OUTPUT
        self.model_config = self.model.config
        
        assert (self.prompt_config.FINAL_OUTPUT in ['CLS','ALL','PROMPTS','SPEECH','PROMPTS+SPEECH']), ('FINAL_OUTPUT parameter not included in the supported configs!')
        assert adapter_type in ['Pfeiffer','Houlsby'], ('Only Pfeiffer and Houlsby adapter is supported for AST!')
        
        
        self.patch_size = _pair(self.model_config.patch_size)
        
        # Prompt Config
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # Prompt PROJECT (No projection for now)
        prompt_dim = self.model_config.hidden_size
        
        self.prompt_proj = nn.Identity()
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
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
        
        
        self._unfreeze_loras()
        self._unfreeze_adapters()
        
        
    def _unfreeze_loras(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].attention.attention.lora_down.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_up.requires_grad_(True)
            
            # Optional: finetune also the LayerNorm befor the MHSA layer.
            #self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
    
    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].adapter_module_FFN.requires_grad_(True)
            self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                self.encoder.layer[block_idx].adapter_module_MHSA.requires_grad_(True)
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
    
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layer[block_idx].adapter_module_FFN.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module_FFN.lnorm.train()
            
                if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
                    self.encoder.layer[block_idx].layernorm_before.train()
                    if self.adapter_config.ADAPTER_BLOCK =='conformer':
                        self.encoder.layer[block_idx].adapter_module_MHSA.bnorm.train()
                        self.encoder.layer[block_idx].adapter_module_MHSA.lnorm.train()
                        
                self.encoder.layer[block_idx].layernorm_after.train()
            
            self.prompt_proj.train()
            self.prompt_dropout.train()
            self.layernorm.train() 
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
        
        
    def incorporate_prompt(self, x):
        B = x.shape[0]
       
        x = self.embeddings(x) 
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
            # print(i)
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
    