#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:51:56 2023

@author: umbertocappellazzo
"""


#################################
#                               #   
#    Code for adapter-tuning.   #
#                               #                                        
#################################


import torch 
import torch.nn as nn
from transformers import ASTModel
from dataclasses import dataclass
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer, ASTEncoder, ASTOutput
from typing import Optional, Tuple, Union


#Here we define the Bottleneck and Convpass adapters.

# BOTTLENECK adapter module.

class Bottleneck_adapter(nn.Module):
    def __init__(self, in_dim, reduction_rate, out_dim):
        super().__init__()
        
        bottleneck_dim = round(in_dim/reduction_rate)
        self.linear_downsample = nn.Linear(in_dim, bottleneck_dim)
        self.linear_upsample = nn.Linear(bottleneck_dim, out_dim)
        #self.layer_norm_adapt = nn.LayerNorm(out_dim)  # If we want to add a LayerNorm after the up-projection layer.
        self.act = torch.nn.GELU()
        
        nn.init.zeros_(self.linear_downsample.weight); nn.init.zeros_(self.linear_upsample.weight)
        nn.init.zeros_(self.linear_downsample.bias); nn.init.zeros_(self.linear_upsample.bias);
        
    def forward(self, x):
        down_x = self.linear_downsample(x)
        up_x = self.linear_upsample(self.act(down_x))
        
        return up_x
        #return self.layer_norm_adapt(up_x)


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


# These are the classes for adapter-tuning. They have been used for the main experiments involving Pfeiffer/Houlsby Bottleneck/Convpass FFN configurations.


@dataclass
class Adapter_config:
    REDUCTION_RATE: int 
    ADAPTER_TYPE: str
    ADAPTER_CONF: str
    APPLY_RESIDUAL: bool
    ADAPTER_BLOCK: str

class ASTModel_adapter(ASTModel):
    def __init__(self, config, adapter_config: Adapter_config):
        super().__init__(config)
        
        self.adapter_config= adapter_config
        
        self.encoder = ASTEncoder_adapter(config, adapter_config)


class ASTEncoder_adapter(ASTEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_adapter(config, adapter_config) for _ in range(config.num_hidden_layers)])
        

# For sequential adapter AST, the ASTOutput does not have to apply residual there.
class ASTOutput_adapter(ASTOutput):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class ASTLayer_adapter(ASTLayer):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        self.adapter_config = adapter_config
        self.config = config
        
        if adapter_config.ADAPTER_CONF == 'sequential':  # * insert not for MIXED.
            self.output = ASTOutput_adapter(config)
        
        self.adapter_module_FFN = self.make_adapter(config.hidden_size,adapter_config.REDUCTION_RATE,config.hidden_size,adapter_config.ADAPTER_BLOCK)
        
        if adapter_config.ADAPTER_TYPE == 'Houlsby':
            self.adapter_module_MHSA = self.make_adapter(config.hidden_size,adapter_config.REDUCTION_RATE,config.hidden_size,adapter_config.ADAPTER_BLOCK)
    
    def make_adapter(self, in_dim, reduction_rate, out_dim, adapter_block):
        
        if adapter_block == 'bottleneck' :
            adapter_layer = Bottleneck_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        elif adapter_block == 'convpass':
            adapter_layer = Convpass_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        else:
            raise Exception('Only convpass and bottleneck adapter modules are supported as of now!')
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in AST, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        
            
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
            
            if self.adapter_config.ADAPTER_CONF == 'parallel':
                output_layernorm = self.layernorm_before(hidden_states)
                adapter_output_MHSA = self.adapter_module_MHSA(output_layernorm)
                if self.adapter_config.APPLY_RESIDUAL:
                    hidden_states = attention_output + hidden_states + adapter_output_MHSA + output_layernorm
                else:
                    hidden_states = attention_output + hidden_states + adapter_output_MHSA
            else:
                adapter_output_MHSA = self.adapter_module_MHSA(attention_output)
                if self.adapter_config.APPLY_RESIDUAL:
                    hidden_states = adapter_output_MHSA + hidden_states + attention_output
                else:
                    hidden_states = adapter_output_MHSA + hidden_states
        else:
        
            hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        
        
                        
        if self.adapter_config.ADAPTER_CONF == 'parallel': #*  == 'sequential' for MIXED.
            
            adapter_output_FFN = self.adapter_module_FFN(layer_output)
            
            intermediate_output = self.intermediate(layer_output)
            if self.adapter_config.APPLY_RESIDUAL:
                layer_output = self.output(intermediate_output, hidden_states) + adapter_output_FFN + layer_output
            else:
                layer_output = self.output(intermediate_output, hidden_states) + adapter_output_FFN
        else:
            intermediate_output = self.intermediate(layer_output)
            ffn_output = self.output(intermediate_output)     
            if self.adapter_config.APPLY_RESIDUAL:  #* insert not for MIXED.
                layer_output = self.adapter_module_FFN(ffn_output) + hidden_states + ffn_output
            else:
                layer_output = self.adapter_module_FFN(ffn_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs
    

class AST_adapter(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        ''' The reduction rate decides the bottleneck dimension of the adapter module --> bottleneck_dim = in_dim/reduction_rate.
            The adapter_type param specifies the type of the adapter. Supported types: "Houlsby" and "Pfeiffer".
            LN_train: whether the LN layers are trained along with the adapters. Original papers train the LNs.
        '''
        
        super().__init__()
        
        self.adapter_config = Adapter_config(reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block)
        self.model = ASTModel_adapter.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
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
        
        self._unfreeze_adapters()
        
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

    def forward_tsne(self,x):
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)
        
        return hidden_states[:,0], hidden_states.mean(dim=1)
    

# Adapter Hydra: it exploits adapters at the FFN level both in parallel and sequential. This is inspired from 
# Sanghyeon Kim et al., "Hydra: Multi-head Low-rank Adaptation for Parameter Efficient Fine-tuning"

@dataclass
class Adapter_config_hydra:
    REDUCTION_RATE: int 
    ADAPTER_BLOCK: str
    ADAPTER_LOCATION: str
    

class ASTModel_adapter_hydra(ASTModel):
    def __init__(self, config, adapter_config_hydra: Adapter_config_hydra):
        super().__init__(config)
        
        self.adapter_config_hydra = adapter_config_hydra
        
        self.encoder = ASTEncoder_adapter_hydra(config, adapter_config_hydra)


class ASTEncoder_adapter_hydra(ASTEncoder):
    def __init__(self, config, adapter_config_hydra):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_adapter_hydra(config, adapter_config_hydra) for _ in range(config.num_hidden_layers)])
        

# For now, the sequential and parallel adapters are inserted only after the FNN module (i.e., only Pfeiffer config supported).
class ASTLayer_adapter_hydra(ASTLayer):
    def __init__(self, config, adapter_config_hydra):
        super().__init__(config)
        
        
        self.output = ASTOutput_adapter(config)
        self.adapter_config_hydra = adapter_config_hydra
        
        self.adapter_module_seq = self.make_adapter(config.hidden_size,adapter_config_hydra.REDUCTION_RATE,config.hidden_size,adapter_config_hydra.ADAPTER_BLOCK)
        self.adapter_module_par = self.make_adapter(config.hidden_size,adapter_config_hydra.REDUCTION_RATE,config.hidden_size,adapter_config_hydra.ADAPTER_BLOCK)
    
    def make_adapter(self, in_dim, reduction_rate, out_dim, adapter_block):
        
        if adapter_block == 'bottleneck' :
            adapter_layer = Bottleneck_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        elif adapter_block == 'convpass':
            adapter_layer = Convpass_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        else:
            raise Exception('Only convpass and bottleneck adapter modules are supported as of now!')
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        
        
        output_LN1 = self.layernorm_before(hidden_states)
        
        if self.adapter_config_hydra.ADAPTER_LOCATION == 'FFN':
            
            self_attention_outputs = self.attention(
                output_LN1,  
                head_mask,
                output_attentions=output_attentions,
            )
            
                
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  
        else:
            self_attention_outputs = self.attention(
                output_LN1,  
                head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  
            
            seq_adapter_output = self.adapter_module_seq(attention_output)
            par_adapter_outout = self.adapter_module_par(output_LN1)
            
            attention_output = attention_output + par_adapter_outout + seq_adapter_output
            
            
        
        
        
        hidden_states = attention_output + hidden_states

        # in AST, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        
        if self.adapter_config_hydra.ADAPTER_LOCATION == 'FFN':
        
            ffn_output = self.output(self.intermediate(layer_output))
            
            seq_adapter_output = self.adapter_module_seq(ffn_output)
            
            paral_adapter_output = self.adapter_module_par(layer_output)
            
            layer_output = hidden_states + ffn_output + seq_adapter_output + paral_adapter_output
        else: 
            ffn_output = self.output(self.intermediate(layer_output))
            layer_output = layer_output+ ffn_output

        outputs = (layer_output,) + outputs

        return outputs
    
    
    
    
class AST_adapter_hydra(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_block: str, adapter_location: str, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        ''' The reduction rate decides the bottleneck dimension of the adapter module --> bottleneck_dim = in_dim/reduction_rate.
            The adapter_type param specifies the type of the adapter. Supported types: "Houlsby" and "Pfeiffer".
            LN_train: whether the LN layers are trained along with the adapters. Original papers train the LNs.
        '''
        
        super().__init__()
        
        self.adapter_config = Adapter_config_hydra(reduction_rate, adapter_block, adapter_location)
        self.model = ASTModel_adapter_hydra.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_adapters()
        
    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].adapter_module_seq.requires_grad_(True)
            self.encoder.layer[block_idx].adapter_module_par.requires_grad_(True)
            if self.adapter_config.ADAPTER_LOCATION == 'FFN':
                self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
            elif self.adapter_config.ADAPTER_LOCATION == 'MHSA':
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
            
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layer[block_idx].adapter_module_seq.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module_seq.lnorm.train()
                    self.encoder.layer[block_idx].adapter_module_par.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module_par.lnorm.train()
                        
                if self.adapter_config.ADAPTER_LOCATION == 'FFN':
                    self.encoder.layer[block_idx].layernorm_after.train()
                if self.adapter_config.ADAPTER_LOCATION == 'MHSA':
                    self.encoder.layer[block_idx].layernorm_before.train()
            
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
        

# These classes are used for the ablation studies on the best configration for the adapter module. 
# It supports: FFN/MHSA insertion, sequential/parallel insertion, and before/after the chosen sub-layer.

@dataclass
class Adapter_config_ablation:
    REDUCTION_RATE: int 
    ADAPTER_seqpar: str  # 'sequential' or 'parallel'
    ADAPTER_BEFAFTER: str # 'before' or 'after' 
    ADAPTER_LOCATION: str # 'MHSA' or 'FFN'
    ADAPTER_BLOCK: str # 'bottleneck' or 'convpass'

class ASTModel_adapter_ablation(ASTModel):
    def __init__(self, config, adapter_config: Adapter_config_ablation):
        super().__init__(config)
        
        self.adapter_config= adapter_config
        
        self.encoder = ASTEncoder_adapter_ablation(config, adapter_config)


class ASTEncoder_adapter_ablation(ASTEncoder):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        self.layer = nn.ModuleList([ASTLayer_adapter_ablation(config, adapter_config) for _ in range(config.num_hidden_layers)])

class ASTLayer_adapter_ablation(ASTLayer):
    def __init__(self, config, adapter_config):
        super().__init__(config)
        
        # This adapter class adds the residual outside the ASTOutput layer.
        self.output = ASTOutput_adapter(config)
        
        self.adapter_module = self.make_adapter(config.hidden_size,adapter_config.REDUCTION_RATE,config.hidden_size,adapter_config.ADAPTER_BLOCK)
        self.adapter_config = adapter_config
        
        
    def make_adapter(self, in_dim, reduction_rate, out_dim, adapter_block):
    
        if adapter_block == 'bottleneck' :
            adapter_layer = Bottleneck_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        elif adapter_block == 'convpass':
            adapter_layer = Convpass_adapter(in_dim, reduction_rate, out_dim)
            return adapter_layer
        else:
            raise Exception('Only convpass and bottleneck are supported as of now!')
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        
        
        output_LN1 = self.layernorm_before(hidden_states)
        
        if self.adapter_config.ADAPTER_LOCATION == 'MHSA':
            
            
            if self.adapter_config.ADAPTER_seqpar == 'parallel':
                if self.adapter_config.ADAPTER_BEFAFTER == 'before':
                    # CASE 1): MHSA-parallel-before
                    
                    adapter_output = self.adapter_module(output_LN1)
                    
                    self_attention_outputs = self.attention(
                        output_LN1,  # in AST, layernorm is applied before self-attention
                        head_mask,
                        output_attentions=output_attentions,
                    )
                    
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
                    
                    mhsa_output = attention_output + adapter_output
                    
                else: #CASE 2): MHSA-parallel-after
                    
                    self_attention_outputs = self.attention(
                        output_LN1,  # in AST, layernorm is applied before self-attention
                        head_mask,
                        output_attentions=output_attentions,
                    )
                    
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
                    
                    adapter_output = self.adapter_module(attention_output)
                    
                    mhsa_output = attention_output + adapter_output
                
                    
            else: # Sequential case
                if self.adapter_config.ADAPTER_BEFAFTER == 'before':
                    # CASE 3): MHSA-sequential-before
                    
                    adapter_output = self.adapter_module(output_LN1)
                    
                    self_attention_outputs = self.attention(
                        adapter_output,  # in AST, layernorm is applied before self-attention
                        head_mask,
                        output_attentions=output_attentions,
                    )
                    
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
                    
                    mhsa_output = attention_output + adapter_output
                
                else: # CASE 4): MHSA-sequential-after:
                    self_attention_outputs = self.attention(
                        output_LN1,  # in AST, layernorm is applied before self-attention
                        head_mask,
                        output_attentions=output_attentions,
                    )
                    
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
                    
                    adapter_output = self.adapter_module(attention_output)
                    
                    mhsa_output = adapter_output + attention_output
                
            
            
            hidden_states = mhsa_output + hidden_states
            
            output_LN2 = self.layernorm_after(hidden_states)
            
            output_up_proj = self.intermediate(output_LN2)
            
            output_down_proj = self.output(output_up_proj) + hidden_states
            
            outputs = (output_down_proj,) + outputs

            return outputs
        
        else:
            
            self_attention_outputs = self.attention(
                output_LN1,  # in AST, layernorm is applied before self-attention
                head_mask,
                output_attentions=output_attentions,
            )
        
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            
            hidden_states = attention_output + hidden_states
            
            output_LN2 = self.layernorm_after(hidden_states)
            
            if self.adapter_config.ADAPTER_seqpar == 'parallel':
                if self.adapter_config.ADAPTER_BEFAFTER == 'before':
                    # CASE 5): FFN-parallel-before
                    output_up_proj = self.intermediate(output_LN2)
                    output_down_proj = self.output(output_up_proj)
                    
                    ffn_output = output_down_proj + self.adapter_module(output_LN2)
            
                else: #CASE 6): FFN-parallel-after
                    output_up_proj = self.intermediate(output_LN2)
                    output_down_proj = self.output(output_up_proj)
                    
                    ffn_output = output_down_proj + self.adapter_module(output_down_proj)
            
            else: # Sequential case
            
                if self.adapter_config.ADAPTER_BEFAFTER == 'before':
                    # CASE 7): FFN-sequential-before
                    
                    adapter_output = self.adapter_module(output_LN2)
                    
                    output_up_proj = self.intermediate(adapter_output)
                    output_down_proj = self.output(output_up_proj)
                    
                    ffn_output = output_down_proj + adapter_output
                    
                else: # CASE 8): FFN-sequential-after:
                    output_up_proj = self.intermediate(output_LN2)
                    output_down_proj = self.output(output_up_proj)
                    
                    adapter_output = self.adapter_module(output_down_proj)
                    
                    ffn_output = output_down_proj + adapter_output

            ffn_output = ffn_output + hidden_states

            outputs = (ffn_output,) + outputs

            return outputs
    
class AST_adapter_ablation(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, seq_or_par: str,  location: str, adapter_block: str, before_after: str, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        ''' The reduction rate decides the bottleneck dimension of the adapter module --> bottleneck_dim = in_dim/reduction_rate.
            The adapter_type param specifies the type of the adapter. Supported types: "Houlsby" and "Pfeiffer".
            LN_train: whether the LN layers are trained along with the adapters. Original papers train the LNs.
        '''
        
        super().__init__()
        
        assert seq_or_par in ['sequential','parallel'], ("Only sequential and parallel are accepted!")
        assert location in ['MHSA','FFN'], ("Only MHSA and FFN are accepted!")
        assert adapter_block in ['convpass','bottleneck','conformer'], ("Only convpass, sequential and conformer are accepted!")
        assert before_after in ['before','after'], ("Only after and before are accepted!")
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.adapter_config = Adapter_config_ablation(reduction_rate, seq_or_par, before_after, location, adapter_block)
        self.model = ASTModel_adapter_ablation.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_adapters()
        
    def _unfreeze_adapters(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].adapter_module.requires_grad_(True)
            if self.adapter_config.ADAPTER_LOCATION == 'MHSA':
                self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
            else: 
                self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
    
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            for block_idx in range(self.model_config.num_hidden_layers):
                
                if self.adapter_config.ADAPTER_BLOCK =='conformer':
                    self.encoder.layer[block_idx].adapter_module.bnorm.train()
                    self.encoder.layer[block_idx].adapter_module.lnorm.train()
            
                if self.adapter_config.ADAPTER_LOCATION == 'MHSA':
                    self.encoder.layer[block_idx].layernorm_before.train()
                else:      
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
