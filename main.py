#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:35:28 2023

@author: umbertocappellazzo
"""

import torch
from torch.optim import AdamW
from src.AST import AST
from src.AST_LoRA import AST_LoRA, AST_LoRA_ablation
from src.AST_adapters import AST_adapter, AST_adapter_ablation
from src.Wav2Vec_adapter import Wav2Vec, Wav2Vec_adapter
from src.AST_prompt_tuning import AST_Prefix_tuning, PromptAST, Prompt_config
from src.MoA import AST_MoA, AST_SoftMoA
from dataset.fluentspeech import FluentSpeech
from dataset.esc_50 import ESC_50
from dataset.urban_sound_8k import Urban_Sound_8k
from dataset.google_speech_commands_v2 import Google_Speech_Commands_v2
from dataset.iemocap import IEMOCAP
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
    parser.add_argument('--seed', default= 10)   #Set it to None if you don't want to set it.
    parser.add_argument('--device', type= str, default= 'cuda', 
                        help='device to use for training/testing')
    parser.add_argument('--num_workers', type= int, default= 4)
    parser.add_argument('--model_ckpt_AST', default= 'MIT/ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--model_ckpt_wav', default='facebook/wav2vec2-base-960h')
    parser.add_argument('--max_len_audio', type= int, default=128000, 
                        help='max length for the audio signal --> it will be cut. Only for IEMOCAP.')
    parser.add_argument('--save_best_ckpt', type= bool, default= False)
    parser.add_argument('--output_path', type= str, default= '/checkpoints')
    parser.add_argument('--is_AST', type= bool, default = False, help ='Whether we are using ther AST model or not (i.e., wav2vec 2.0 etc.).')
    
    parser.add_argument('--dataset_name', type= str, default= 'IEMOCAP', choices = ['FSC', 'ESC-50', 'urbansound8k', 'GSC', 'IEMOCAP'])
    parser.add_argument('--method', type= str, default='adapter',choices = ['linear', 'full-FT', 'adapter', 'prompt-tuning', 'prefix-tuning', 
                                                          'LoRA', 'BitFit', 'Dense-MoA', 'Soft-MoA'])
    # Adapter params.
    parser.add_argument('--seq_or_par', default = 'parallel', choices=['sequential','parallel'])
    parser.add_argument('--reduction_rate_adapter', type= int, default= 64)
    parser.add_argument('--adapter_type', type= str, default = 'Pfeiffer', choices = ['Houlsby', 'Pfeiffer'])
    parser.add_argument('--apply_residual', type= bool, default=False)
    parser.add_argument('--adapter_block', type= str, default='conformer', choices = ['bottleneck', 'conformer'])
    parser.add_argument('--kernel_size', type= int, default=31,)

    # Params for adapter ablation studies.
    parser.add_argument('--is_adapter_ablation', default= False)
    parser.add_argument('--befafter', type = str, default='after', choices = ['after','before'])
    parser.add_argument('--location', type = str, default='FFN', choices = ['MHSA','FFN'])
    
    # Soft/Dense MoA params.
    parser.add_argument('--reduction_rate_moa', type= int, default= 128) 
    parser.add_argument('--adapter_type_moa', type= str, default= 'Pfeiffer', choices = ['Houlsby', 'Pfeiffer'])
    parser.add_argument('--location_moa', type = str, default='MHSA', choices = ['MHSA','FFN'])
    parser.add_argument('--adapter_module_moa', type= str, default= 'bottleneck', choices = ['bottleneck', 'conformer'])
    parser.add_argument('--num_adapters', type= int, default= 7)
    parser.add_argument('--num_slots', type = int, default= 1)
    parser.add_argument('--normalize', type = bool, default= False)
    
    
    # LoRA params.
    parser.add_argument('--reduction_rate_lora', type= int, default= 64)
    parser.add_argument('--alpha_lora', type= int, default= 8)

    # Params for LoRA ablation studies.
    parser.add_argument('--is_lora_ablation', type= bool, default= False)
    parser.add_argument('--lora_config', type = str, default = 'Wq,Wv', choices = ['Wq','Wq,Wk','Wq,Wv','Wq,Wk,Wv,Wo'])
    
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
    parser.add_argument('--project_name', type= str, default= '')
    parser.add_argument('--exp_name', type= str, default= '')
    parser.add_argument('--entity', type= str, default= '')
    
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
        max_len_AST = train_params['max_len_AST_FSC']
        num_classes = train_params['num_classes_FSC']
        batch_size = train_params['batch_size_FSC']
        epochs = train_params['epochs_FSC_AST'] if args.is_AST else train_params['epochs_FSC_WAV']
    elif args.dataset_name == 'ESC-50':
        max_len_AST = train_params['max_len_AST_ESC']
        num_classes = train_params['num_classes_ESC']
        batch_size = train_params['batch_size_ESC']
        epochs = train_params['epochs_ESC']
    elif args.dataset_name == 'urbansound8k':
        max_len_AST = train_params['max_len_AST_US8K']
        num_classes = train_params['num_classes_US8K']
        batch_size = train_params['batch_size_US8K']
        epochs = train_params['epochs_US8K']
    elif args.dataset_name == 'GSC':
        max_len_AST = train_params['max_len_AST_GSC']
        num_classes = train_params['num_classes_GSC']
        batch_size = train_params['batch_size_GSC']
        epochs = train_params['epochs_GSC_AST'] if args.is_AST else train_params['epochs_GSC_WAV']
    elif args.dataset_name == 'IEMOCAP':
        max_len_AST = train_params['max_len_AST_IEMO']
        num_classes = train_params['num_classes_IEMO']
        batch_size = train_params['batch_size_IEMO']
        epochs = train_params['epochs_IEMO']
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
    elif args.dataset_name == 'urbansound8k':
        fold_number = 10
        folds_train = [[1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10], [3,4,5,6,7,8,9,10,1], 
                       [4,5,6,7,8,9,10,1,2], [5,6,7,8,9,10,1,2,3], [6,7,8,9,10,1,2,3,4], 
                       [7,8,9,10,1,2,3,4,5], [8,9,10,1,2,3,4,5,6], [9,10,1,2,3,4,5,6,7], [10,1,2,3,4,5,6,7,8]]
        folds_test = [[10], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    else:
        assert args.dataset_name == 'IEMOCAP'
        fold_number = 10
        sessions_train = [[1,2,3,4], [1,2,3,4], [2,3,4,5], [2,3,4,5], [3,4,5,1], [3,4,5,1], [4,5,1,2], [4,5,1,2], [5,1,2,3], [5,1,2,3]]
        session_val = [[5], [5], [1], [1], [2], [2], [3], [3], [4], [4]]
        speaker_id_val = ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
        speaker_id_test = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']

    
    for fold in range(0,fold_number):
        
        # DATASETS
        
        if args.dataset_name == 'FSC':
            train_data = FluentSpeech(args.data_path, max_len_AST, train= True, apply_SpecAug= False, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = FluentSpeech(args.data_path, max_len_AST, train= "valid")
            test_data = FluentSpeech(args.data_path, max_len_AST, train= False)
        
        elif args.dataset_name == 'ESC-50': 
            train_data = ESC_50(args.data_path, max_len_AST, 'train', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold], apply_SpecAug= True, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = ESC_50(args.data_path, max_len_AST, 'valid', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
            test_data = ESC_50(args.data_path, max_len_AST, 'test', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
        elif args.dataset_name == 'urbansound8k':
            train_data = Urban_Sound_8k(args.data_path, max_len_AST, 'train', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold], apply_SpecAug=True, few_shot=args.is_few_shot_exp, samples_per_class= args.few_shot_samples,)
            test_data = Urban_Sound_8k(args.data_path, max_len_AST, 'test', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold])
        elif args.dataset_name == 'GSC':
            train_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'train', apply_SpecAug= False, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
            val_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'valid')
            test_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'test')
        else:
            train_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= sessions_train[fold], speaker_id= 'both', is_AST = args.is_AST, apply_SpecAug=False, few_shot=args.is_few_shot_exp, samples_per_class=args.few_shot_samples)
            val_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= session_val[fold], speaker_id= speaker_id_val[fold], is_AST = args.is_AST)
            test_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= session_val[fold], speaker_id= speaker_id_test[fold], is_AST = args.is_AST)
        
        train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
        test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
        
        if args.dataset_name != 'urbansound8k': # US8K does not have the validation set.
            val_loader = DataLoader(val_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
            
            
        # MODEL definition.
        
        method = args.method
        
        if args.is_AST:

            if args.is_adapter_ablation:
                model = AST_adapter_ablation(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, seq_or_par= args.seq_or_par, location= args.location, adapter_block= args.adapter_block, before_after= args.befafter, kernel_size = args.kernel_size, model_ckpt= args.model_ckpt_AST).to(device)
                lr= train_params['lr_adapter']
            elif args.is_lora_ablation:
                model = AST_LoRA_ablation(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, lora_config= args.lora_config, model_ckpt= args.model_ckpt_AST).to(device)
                lr = train_params['lr_LoRA']
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
                model = AST_adapter(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_adapter, adapter_type= args.adapter_type, seq_or_par= args.seq_or_par, apply_residual= args.apply_residual, adapter_block= args.adapter_block, kernel_size =args.kernel_size, model_ckpt= args.model_ckpt_AST).to(device)
                lr = train_params['lr_adapter']
            elif method == 'Dense-MoA':
                model = AST_MoA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_moa, adapter_type= args.adapter_type_moa, location= args.location_moa, adapter_module= args.adapter_module_moa, num_adapters= args.num_adapters, model_ckpt= args.model_ckpt_AST).to(device)
                lr = train_params['lr_MoA']
            elif method == 'Soft-MoA':
                model = AST_SoftMoA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, reduction_rate= args.reduction_rate_moa, adapter_type= args.adapter_type_moa, location= args.location_moa, adapter_module= args.adapter_module_moa, num_adapters= args.num_adapters, num_slots= args.num_slots, normalize= args.normalize, model_ckpt= args.model_ckpt_AST).to(device)
                lr = train_params['lr_MoA']
            else:
                raise ValueError('The method you chose is not supported as of now.')
        else:
            if method == 'full-FT':
                model = Wav2Vec(num_classes = num_classes, model_ckpt = args.model_ckpt_wav).to(device)
                # Only the last 3 layers of Wav2vec are fine-tuned. Fine-tuning all 12 layers results in complete overfitting.
                
                model.model.requires_grad_(False)
                model.model.encoder.layers[-1].requires_grad_(True)
                model.model.encoder.layers[-2].requires_grad_(True)
                model.model.encoder.layers[-3].requires_grad_(True)
                lr = train_params['lr_WAV_fullFT']
            elif method == 'linear': 
                model = Wav2Vec(num_classes = num_classes, model_ckpt = args.model_ckpt_wav).to(device)
                # Freeze the encoder, only the classifier is trainable.
                model.model.requires_grad_(False)
                model.model_config.apply_spec_augment = False
                lr = train_params['lr_WAV_linear']
            elif method == 'adapter':
                model = Wav2Vec_adapter(num_classes, args.reduction_rate_adapter, args.adapter_type, args.seq_or_par, args.apply_residual, args.adapter_block, kernel_size =args.kernel_size, model_ckpt= args.model_ckpt_wav).to(device)
                lr = train_params['lr_WAV_adapter']
                
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
                    torch.save(best_params, os.getcwd() + args.output_path + f'/bestmodel_fold{fold}')   
            
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
    print(f"Avg accuracy over the {fold_number} fold(s): ", np.mean(accuracy_folds))
    print(f"Std accuracy over the {fold_number} fold(s): ", np.std(accuracy_folds))
    
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