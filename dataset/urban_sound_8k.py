#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:39:11 2023

@author: umbertocappellazzo
"""

import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor
import torchaudio, torch


class Urban_Sound_8k(Dataset):
    """
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: 
        air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, 
        siren, and street_music.
    """

    def __init__(self, data_path, max_len_audio, max_len_AST,split, train_fold_nums=[1,2,3,4,5,6,7,8,9], test_fold_nums=[10], apply_SpecAug= False, few_shot = False, samples_per_class = 1):
        if split not in ("train", "test"):
            raise ValueError(f"`train` arg ({split}) must be a bool or train/test.")
            
        self.data_path = os.path.expanduser(data_path)
        self.max_len_audio = max_len_audio
        self.max_len_AST = max_len_AST
        self.split = split
        self.train_fold_nums = train_fold_nums
        self.test_fold_nums = test_fold_nums
        
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
        
        if self.split == 'train': 
            fold = self.train_fold_nums
        else:
            fold = self.test_fold_nums
        
        
        processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST)
        
        x, y = [], []

        with open(os.path.join(self.data_path, "UrbanSound8K/metadata","UrbanSound8K.csv")) as f:
            lines = f.readlines()[1:]
        
        
        for line in lines:
            
            items = line[:-1].split(',')
            
            if int(items[-3]) not in fold: continue  # The sample is not included in the fold we are interested in.

            # Read the wav audio, apply processor depending on the SS model and store it in the overall list.
            
            pathh = os.path.join(self.data_path, 'UrbanSound8K/audio', 'fold'+items[-3], items[0])
            wav,sampling_rate = soundfile.read(pathh)
            
            if len(wav.shape) > 1: wav = wav[:,0] 
            
            # Resample the sampling frequency from 44.1kHz to 16kHz.
            wav = librosa.resample(wav, orig_sr =sampling_rate, target_sr = 16000)
            
            x.append(processor(wav, sampling_rate= 16000, return_tensors='pt')['input_values'].squeeze(0))
                
            y.append(
                self.class_ids[items[-1]]    
            )
            
        return np.array(x), np.array(y)
    
    
    @property
    def class_ids(self):
        return {
             'air_conditioner': 0,
             'car_horn': 1,
             'children_playing': 2,
             'dog_bark': 3,
             'drilling': 4,
             'engine_idling': 5,
             'gun_shot': 6,
             'jackhammer': 7,
             'siren': 8,
             'street_music': 9,
        }
    