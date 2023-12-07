#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:12:02 2022

@author: umbertocappellazzo
"""

import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor
import torchaudio, torch


class ESC_50(Dataset):
    """
    The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings 
    suitable for benchmarking methods of environmental sound classification.
    The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class), 
    loosely arranged into 5 major categories. 
    """

    def __init__(self, data_path, max_len_AST, split, train_fold_nums=[1, 2, 3], valid_fold_nums=[4], test_fold_nums=[5], apply_SpecAug= False, few_shot = False, samples_per_class = 1):
        if split not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({split}) must be a bool or train/valid/test.")
            
        self.data_path = os.path.expanduser(data_path)
        self.max_len_AST = max_len_AST
        self.split = split
        self.train_fold_nums = train_fold_nums
        self.valid_fold_nums = valid_fold_nums
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
        elif self.split == 'valid':
            fold = self.valid_fold_nums
        else:
            fold = self.test_fold_nums
        
        processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST)
        
        x, y = [], []

        with open(os.path.join(self.data_path, "ESC-50/meta","esc50.csv")) as f:
            lines = f.readlines()[1:]
        
        
        for line in lines:
            
            items = line[:-1].split(',')
            
            if int(items[1]) not in fold: continue  # The sample is not included in the fold we are interested in.

            # Read the wav audio, apply processor depending on the SS model and store it in the overall list.
            
            pathh = os.path.join(self.data_path, 'ESC-50/audio', items[0])
            wav,sampling_rate = soundfile.read(pathh)
            # Resample the sampling frequency from 44.1kHz to 16kHz.
            wav = librosa.resample(wav, orig_sr =sampling_rate, target_sr = 16000)
            
            
            x.append(processor(wav, sampling_rate= 16000, return_tensors='pt')['input_values'].squeeze(0))
                
            y.append(
                self.class_ids[items[3]]    
            )
            
        return np.array(x), np.array(y)
    
    @property
    def class_ids(self):
        return {
             'dog': 0,
             'rooster': 1,
             'pig': 2,
             'cow': 3,
             'frog': 4,
             'cat': 5,
             'hen': 6,
             'insects': 7,
             'sheep': 8,
             'crow': 9,
             'rain': 10,
             'sea_waves': 11,
             'crackling_fire': 12,
             'crickets': 13,
             'chirping_birds': 14,
             'water_drops': 15,
             'wind': 16,
             'pouring_water': 17,
             'toilet_flush': 18,
             'thunderstorm': 19,
             'crying_baby': 20,
             'sneezing': 21,
             'clapping': 22,
             'breathing': 23,
             'coughing': 24,
             'footsteps': 25,
             'laughing': 26,
             'brushing_teeth': 27,
             'snoring': 28,
             'drinking_sipping': 29,
             'door_wood_knock': 30,
             'mouse_click': 31,
             'keyboard_typing': 32,
             'door_wood_creaks': 33,
             'can_opening': 34,
             'washing_machine': 35,
             'vacuum_cleaner': 36,
             'clock_alarm': 37,
             'clock_tick': 38,
             'glass_breaking': 39,
             'helicopter': 40,
             'chainsaw': 41,
             'siren': 42,
             'car_horn': 43,
             'engine': 44,
             'train': 45,
             'church_bells': 46,
             'airplane': 47,
             'fireworks': 48,
             'hand_saw': 49,
        }