#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:41:17 2023

@author: umbertocappellazzo
"""

import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor
import torchaudio, torch


class Google_Speech_Commands_v2(Dataset):
    """
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: 
        air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, 
        siren, and street_music.
    """

    def __init__(self, data_path, max_len_audio, max_len_AST, split, apply_SpecAug= False, few_shot = False, samples_per_class = 1):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({split}) must be a bool or train/valid/test.")
            
        self.data_path = os.path.expanduser(data_path)
        self.max_len_audio = max_len_audio
        self.max_len_AST = max_len_AST
        self.split = split
        
        self.apply_SpecAug = apply_SpecAug
        self.freq_mask = 24
        self.time_mask = 80
        
        self.x, self.y = self.get_data()
        
        if few_shot:
            self.x, self.y = self.get_few_shot_data(samples_per_class)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        
        if self.apply_SpecAug:
        
            freqm = torchaudio.transforms.FrequencyMasking(self.freq_mask)
            timem = torchaudio.transforms.TimeMasking(self.time_mask)
            
            fbank = torch.transpose(self.x[index], 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
            
            return fbank, self.y[index]
        else:
            return self.x[index], self.y[index]
    
    
    def get_few_shot_data(self, samples_per_class: int):
        x_few, y_few = [], []
        
        total_classes = np.unique(self.y)
        
        for class_ in total_classes:
            cap = 0
            
            for index in range(len(self.y)):
                if self.y[index] == class_:
                    x_few.append(self.x[index])
                    y_few.append(self.y[index])
                    
                    cap += 1
                    if cap == samples_per_class: break
        return x_few, y_few
    
    def get_data(self):
        
        
        if self.split == 'valid': 
            list_name = 'validation_list.txt'
        elif self.split == 'test':
            list_name = 'testing_list.txt'
        else: # train needs both lists.
            list_test_name = 'testing_list.txt'; list_valid_name = 'validation_list.txt'
        
        
        processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST,) 
        
        
        x, y = [], []
        
        if self.split in ['valid','test']:
            
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_name)) as f:
                lines = f.readlines()
            
            for line in lines:
            
                pathh = os.path.join(self.data_path, 'speech_commands_v0.02', line.strip())
                wav,sampling_rate = soundfile.read(pathh)
                
                
                x.append(processor(wav, sampling_rate= 16000, return_tensors='pt')['input_values'].squeeze(0))
                    
                
                y.append(
                    self.class_ids[line.split('/')[0]]    
                )
                
            return np.array(x), np.array(y)
        
        else:
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_valid_name)) as f:
                lines_valid = f.readlines(); lines_valid = [x.strip() for x in lines_valid]
            with open(os.path.join(self.data_path, 'speech_commands_v0.02', list_test_name)) as f:
                lines_test = f.readlines(); lines_test = [x.strip() for x in lines_test]
            
            for class_id in self.class_ids:
                list_files = os.listdir(os.path.join(self.data_path, 'speech_commands_v0.02',class_id))
                
                for file_class in list_files:
                    file_class_ = class_id+'/'+file_class
                   
                    if file_class_ in lines_valid or file_class_ in lines_test:
                        continue
                    
                    pathh = os.path.join(self.data_path, 'speech_commands_v0.02', class_id, file_class)
                    wav,sampling_rate = soundfile.read(pathh)
                    
                    x.append(processor(wav, sampling_rate= 16000, return_tensors='pt')['input_values'].squeeze(0))

                    y.append(
                        self.class_ids[class_id]    
                        )
                
            return np.array(x), np.array(y)
    @property
    def class_ids(self):
        return {
             'backward': 0,
             'bed': 1,
             'bird': 2,
             'cat': 3,
             'dog': 4,
             'down': 5,
             'eight': 6,
             'five': 7,
             'follow': 8,
             'forward': 9,
             'four': 10,
             'go': 11,
             'happy': 12,
             'house': 13,
             'learn': 14,
             'left': 15,
             'marvin': 16,
             'nine': 17,
             'no': 18,
             'off': 19,
             'on': 20,
             'one': 21,
             'right': 22,
             'seven': 23,
             'sheila': 24,
             'six': 25,
             'stop': 26,
             'three': 27,
             'tree': 28,
             'two': 29,
             'up': 30,
             'visual': 31,
             'wow': 32,
             'yes': 33,
             'zero': 34,
             
        }
    