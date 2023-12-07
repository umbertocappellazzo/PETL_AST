#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:31:46 2023

@author: umbertocappellazzo
"""

from transformers import ASTModel
import torch.nn as nn


#Code for full fine-tuning/linear probing.

class AST(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
    def forward(self, x):
        
        hidden_states = self.model(x)[0]
        
        if self.final_output == 'CLS':
            return self.classification_head(hidden_states[:,0])
        else:
            return self.classification_head(hidden_states.mean(dim=1))
    
    def forward_tsne(self,x):
        hidden_states = self.model(x)[0]
        return hidden_states[:,0], hidden_states.mean(dim=1)