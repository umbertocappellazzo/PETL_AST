#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:35:28 2023

@author: umbertocappellazzo
"""

import torch
from torch.optim import AdamW
from src.AST import AST
from src.AST_LoRA import AST_LoRA
from src.AST_adapters import AST_adapter, AST_adapter_hydra, AST_adapter_ablation
from src.AST_prompt_tuning import AST_Prefix_tuning, PromptAST, Prompt_config
from src.PETL_combination import AST_adapterPrompt, AST_LoRA_Adapter, AST_LoRA_Adapter_Prompt
from dataset.fluentspeech import FluentSpeech
from dataset.esc_50 import ESC_50
from dataset.urban_sound_8k import Urban_Sound_8k
from dataset.google_speech_commands_v2 import Google_Speech_Commands_v2
from utils.engine import eval_one_epoch, train_one_epoch
from torch.utils.data import DataLoader
import wandb
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import datetime
import yaml
import os
import copy

def get_args_parser():
    parser = argparse.ArgumentParser('Parameter-efficient Transfer-learning of AST',
                                     add_help=False)
    parser.add_argument('--data_path', type= str, help = 'Path to the location of the dataset.')
    parser.add_argument('--seed', default= 10)   #10 #19   Set it to None if you don't want to set it.
    parser.add_argument('--device', type= str, default= 'cuda', 
                        help='device to use for training/testing')
    parser.add_argument('--num_workers', type= int, default= 4)
    parser.add_argument('--model_ckpt_AST', default= 'MIT/ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--save_best_ckpt', type= bool, default= False)
    parser.add_argument('--output_path', type= str, default= '/checkpoints')
    
    parser.add_argument('--dataset_name', type= str, choices = ['FSC', 'ESC-50', 'urbansound8k', 'GSC'])
    parser.add_argument('--method', type= str, choices = ['linear', 'full-FT', 'adapter', 'prompt-tuning', 'prefix-tuning', 'LoRA', 'BitFit', 'adapter-Hydra', 'adapter+LoRA', 'adapter+prompt', 'adapter+prompt+LoRA'])
    
    # Adapter params.
    parser.add_argument('--seq_or_par', default = 'parallel', choices=['sequential','parallel'])
    parser.add_argument('--reduction_rate_adapter', type= int, default= 64)
    parser.add_argument('--adapter_type', type= str, default = 'Pfeiffer', choices = ['Houlsby', 'Pfeiffer'])
    parser.add_argument('--apply_residual', type= bool, default=False)
    parser.add_argument('--adapter_block', type= str, default='bottleneck', choices = ['bottleneck', 'convpass'])
    
    # Adapter Hydra params.
    parser.add_argument('--location_hydra', type = str, default='FFN', choices = ['MHSA','FFN'])
    
    # Params for adapter ablation studies.
    parser.add_argument('--is_adapter_ablation', default= False)
    parser.add_argument('--befafter', type = str, default='after', choices = ['after','before'])
    parser.add_argument('--location', type = str, default='FFN', choices = ['MHSA','FFN'])
    
    
    # LoRA params.
    parser.add_argument('--reduction_rate_lora', type= int, default= 64)
    parser.add_argument('--alpha_lora', type= int, default= 8)
    
    # Prefix-tuning params.
    parser.add_argument('--prompt_len_pt', type= int, default =24)
    
    # Prompt-tuning params.
    parser.add_argument('--prompt_len_prompt', type= int, default = 25)
    parser.add_argument('--is_deep_prompt', type= bool, default= True)
    parser.add_argument('--drop_prompt', default= 0.)
    
    # Few-shot experiments.
    parser.add_argument('--is_few_shot_exp', default = False)
    parser.add_argument('--few_shot_samples', default = 64)
    
    # WANDB args. 
    parser.add_argument('--use_wandb', type= bool, default= False)
    parser.add_argument('--project_name', type= str, default= 'Prompt-tuning FluentSpeech')
    parser.add_argument('--exp_name', type= str, default= 'prova_codice_adapter_ablation_FFN-after-parallel_seed10_GPU1')
    parser.add_argument('--entity', type= str, default= 'umbertocappellazzo')
    
    return parser

def main(args):
    
    start_time = time.time()
    
    if args.use_wandb:
        wandb.init(project= args.project_name, name= args.exp_name,  entity= args.entity,
                   )
        print(args) 
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    device = torch.device(args.device)
    
    # Fix the seed for reproducibility (if desired).
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed) 
    
    with open('hparams/train.yaml', 'r') as file:
        train_params = yaml.safe_load(file)
    
    if args.dataset_name == 'FSC':
        max_len_audio = train_params['max_len_audio_FSC']
        max_len_AST = train_params['max_len_AST_FSC']
        num_classes = train_params['num_classes_FSC']
        batch_size = train_params['batch_size_FSC']
        epochs = train_params['epochs_FSC']
    elif args.dataset_name == 'ESC-50':
        max_len_audio = train_params['max_len_audio_ESC']
        max_len_AST = train_params['max_len_AST_ESC']
        num_classes = train_params['num_classes_ESC']
        batch_size = train_params['batch_size_ESC']
        epochs = train_params['epochs_ESC']
    elif args.dataset_name == 'urbansound8k':
        max_len_audio = train_params['max_len_audio_US8K']
        max_len_AST = train_params['max_len_AST_US8K']
        num_classes = train_params['num_classes_US8K']
        batch_size = train_params['batch_size_US8K']
        epochs = train_params['epochs_US8K']
    elif args.dataset_name == 'GSC':
        max_len_audio = train_params['max_len_audio_GSC']
        max_len_AST = train_params['max_len_AST_GSC']
        num_classes = train_params['num_classes_GSC']
        batch_size = train_params['batch_size_GSC']
        epochs = train_params['epochs_GSC']
    else:
        raise ValueError('The dataset you chose is not supported as of now.')
        
    
    if args.method == 'prompt-tuning':
        final_output = train_params['final_output_prompt_tuning']
    else:
        final_output = train_params['final_output']
    
    
    accuracy_folds = []
    
    if args.dataset_name in ['FSC', 'GSC']:
        fold_number = 1
    elif args.dataset_name == 'ESC-50':
        fold_number = 5
        folds_train = [[1,2,3], [2,3,4], [3,4,5], [4,5,1], [5,1,2]]
        folds_valid = [[4], [5], [1], [2], [3]]
        folds_test = [[5], [1], [2], [3], [4]]
    else:
        fold_number = 10
        folds_train = [[1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10], [3,4,5,6,7,8,9,10,1], 
                       [4,5,6,7,8,9,10,1,2], [5,6,7,8,9,10,1,2,3], [6,7,8,9,10,1,2,3,4], 
                       [7,8,9,10,1,2,3,4,5], [8,9,10,1,2,3,4,5,6], [9,10,1,2,3,4,5,6,7], [10,1,2,3,4,5,6,7,8]]
        folds_test = [[10], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    
    for fold in range(1,fold_number+1):
        
        # DATASETS
        
        if args.dataset_name == 'FSC':
            train_data = FluentSpeech(args.data_path, max_len_audio, max_len_AST, train= True, apply_SpecAug= True, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = FluentSpeech(args.data_path, max_len_audio, max_len_AST, train= "valid")
            test_data = FluentSpeech(args.data_path, max_len_audio, max_len_AST, train= False)
        
        elif args.dataset_name == 'ESC-50': 
            train_data = ESC_50(args.data_path, max_len_audio, max_len_AST, 'train', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold], apply_SpecAug= True, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = ESC_50(args.data_path, max_len_audio, max_len_AST, 'valid', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
            test_data = ESC_50(args.data_path, max_len_audio, max_len_AST, 'test', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
        elif args.dataset_name == 'urbansound8k':
            train_data = Urban_Sound_8k(args.data_path, max_len_audio, max_len_AST, 'train', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold], apply_SpecAug=True, few_shot=args.is_few_shot_exp, samples_per_class= args.few_shot_samples,)
            test_data = Urban_Sound_8k(args.data_path, max_len_audio, max_len_AST, 'test', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold])
        else:
            train_data = Google_Speech_Commands_v2(args.data_path, max_len_audio, max_len_AST, 'train', apply_SpecAug= True, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = Google_Speech_Commands_v2(args.data_path, max_len_audio, max_len_AST, 'valid')
            test_data = Google_Speech_Commands_v2(args.data_path, max_len_audio, max_len_AST, 'test')
    
        
        train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
        test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
        
        if args.dataset_name != 'urbansound8k': # US8K does not have the validation set.
            val_loader = DataLoader(val_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
            
            
        # MODEL definition.
        
        method = args.method
        
        if args.is_adapter_ablation:
            model = AST_adapter_ablation(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, seq_or_par= args.seq_or_par, location= args.location, adapter_block= args.adapter_block, before_after= args.befafter, model_ckpt= args.model_ckpt_AST).to(device)
            lr= train_params['lr_adapter']
        elif method == 'full-FT':
            model = AST(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_fullFT']
        elif method == 'linear':
            model = AST(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, model_ckpt= args.model_ckpt_AST).to(device)
            # Freeze the AST encoder, only the classifier is trainable.
            model.model.requires_grad_(False)
            # LN is trainable.
            model.model.layernorm.requires_grad_(True)
            lr = train_params['lr_linear']
        elif method == 'BitFit':
            model = AST(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, model_ckpt= args.model_ckpt_AST).to(device)
            model.model.requires_grad_(False)
            for module in model.model.modules():
                if isinstance(module,torch.nn.Linear) or isinstance(module,torch.nn.LayerNorm):
                    module.bias.requires_grad_(True)
            lr = train_params['lr_FitBit']
        elif method == 'LoRA':
            model = AST_LoRA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_LoRA']
        elif method == 'prefix-tuning':
            model = AST_Prefix_tuning(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, num_tokens= args.prompt_len_pt, patch_size= train_params['patch_size'], hidden_size= train_params['hidden_size'], model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_prompt']
        elif method == 'prompt-tuning':
            prompt_config = Prompt_config(NUM_TOKENS= args.prompt_len_prompt, DEEP= args.is_deep_prompt, DROPOUT= args.drop_prompt, FINAL_OUTPUT=final_output)
            model = PromptAST(prompt_config= prompt_config, max_length= max_len_AST, num_classes= num_classes, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_prompt']
        elif method == 'adapter':
            model = AST_adapter(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
        elif method == 'adapter-Hydra':
            model = AST_adapter_hydra(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_block= args.adapter_block, adapter_location= args.location_hydra, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
        elif method == 'adapter+LoRA':
            model = AST_LoRA_Adapter(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
        elif method == 'adapter+prompt':
            prompt_config = Prompt_config(NUM_TOKENS= args.prompt_len_prompt, DEEP= args.is_deep_prompt, DROPOUT= args.drop_prompt, FINAL_OUTPUT=final_output)
            model = AST_adapterPrompt(prompt_config= prompt_config, max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate = args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
        elif method == 'adapter+prompt+LoRA':
            prompt_config = Prompt_config(NUM_TOKENS= args.prompt_len_prompt, DEEP= args.is_deep_prompt, DROPOUT= args.drop_prompt, FINAL_OUTPUT=final_output)
            model = AST_LoRA_Adapter_Prompt(prompt_config= prompt_config, max_length= max_len_AST , num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
            lr = train_params['lr_adapter']
        else:
            raise ValueError('The method you chose is not supported as of now.')
            
        
        # PRINT MODEL PARAMETERS
        n_parameters = sum(p.numel() for p in model.parameters())
        print('Number of params of the model:', n_parameters)
        
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print('Number of trainable params of the model:', n_parameters)
        
        print(model)
        
        
        if method == 'linear': # LR of the backbone to finetune must be quite smaller than the classifier.
            optimizer = AdamW([{'params': model.model.parameters()}, {'params': model.classification_head.parameters(),'lr': 1e-3}],lr= lr,
                                  betas= (0.9,0.98), eps= 1e-6, weight_decay= train_params['weight_decay'] )
        else:
            optimizer = AdamW(model.parameters(), lr= lr ,betas= (0.9,0.98),eps= 1e-6, weight_decay= train_params['weight_decay'])
            
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*(epochs))

        print(f"Start training for {epochs} epochs")
        
        best_acc = 0.
        
        for epoch in range(epochs):
            train_loss, train_acc= train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
            print(f"Trainloss at epoch {epoch}: {train_loss}")
           
            if args.dataset_name == 'urbansound8k':
                val_loss, val_acc = eval_one_epoch(model, test_loader, device, criterion)
            else:
                val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = model.state_dict()
                
                if args.save_best_ckpt:
                    torch.save(best_params, os.getcwd() + args.output_path)   
            
            print("Train intent accuracy: ", train_acc*100)
            print("Valid intent accuracy: ", val_acc*100)           
           
            current_lr = optimizer.param_groups[0]['lr']
            print('Learning rate after initialization: ', current_lr)
            
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, "valid_loss": val_loss,
                           "train_accuracy": train_acc, "val_accuracy": val_acc,
                           "lr": current_lr, }
                          )
        
        best_model = copy.copy(model)
        best_model.load_state_dict(best_params)
        
        test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)
        
        accuracy_folds.append(test_acc)
        
    
    print("Folds accuracy: ", accuracy_folds)
    print("Avg accuracy over the 10 folds: ", np.mean(accuracy_folds))
    print("Std accuracy over the 10 folds: ", np.std(accuracy_folds))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.use_wandb:
        wandb.finish()

if __name__=="__main__":
    parser = argparse.ArgumentParser('Parameter-efficient Transfer-learning of AST',
                                    parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)