#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:12:02 2022

@author: umbertocappellazzo
"""

import os
import torchaudio, torch
from typing import Union
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
from torchaudio import transforms as t
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor



class FluentSpeech(Dataset):
    """
    FSC includes 30,043 English utterances, recorded at 16 kHz.
    It includes 31 intent classes in total.
    """
    def __init__(self, data_path, max_len_AST, train: Union[bool, str] = True, apply_SpecAug= False, few_shot = False, samples_per_class = 1):
        if not isinstance(train, bool) and train not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test.")
            
        if isinstance(train, bool):
            if train:
                self.train = "train"
            else:
                self.train = "test"
        if train in ("train", "valid", "test"):
            self.train = train
        self.max_len_AST = max_len_AST
        self.data_path = os.path.expanduser(data_path)
        
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

    
    def get_tsne_data(self, samples_per_class: int, desired_classes: list):
        x_tsne, y_tsne = [], []
        
        desired_classes = np.array(desired_classes)
        
        for class_ in desired_classes:
            cap = 0
            
            for index in range(len(self.y)):
                if self.y[index] == class_:
                    x_tsne.append(self.x[index])
                    y_tsne.append(self.y[index])
                    
                    cap += 1
                    if cap == samples_per_class: break
        self.x, self.y = x_tsne, y_tsne
        
        
        

    def get_data(self):
        processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST)
        
        base_path = os.path.join(self.data_path, "fluent_speech_commands_dataset")

        x, y= [], []

        with open(os.path.join(base_path, "data", f"{self.train}_data.csv")) as f:
            lines = f.readlines()[1:]

        for line in lines:
            items = line[:-1].split(',')

            action, obj, location = items[-3:]
            pathh = os.path.join(base_path, items[1])
            wav,sampling_rate = soundfile.read(pathh)

            
            
            x.append(processor(wav, sampling_rate= sampling_rate, return_tensors='pt')['input_values'].squeeze(0))
                
            y.append(
                self.class_ids[action+obj+location]    
            )
            
        return np.array(x), np.array(y)
    
    @property
    def transformations(self):
        return None
    
    @property
    def class_ids(self):
        return {
             'change languagenonenone': 0,
             'activatemusicnone': 1,
             'activatelightsnone': 2,
             'deactivatelightsnone': 3,
             'increasevolumenone': 4,
             'decreasevolumenone': 5,
             'increaseheatnone': 6,
             'decreaseheatnone': 7,
             'deactivatemusicnone': 8,
             'activatelampnone': 9,
             'deactivatelampnone': 10,
             'activatelightskitchen': 11,
             'activatelightsbedroom': 12,
             'activatelightswashroom': 13,
             'deactivatelightskitchen': 14,
             'deactivatelightsbedroom': 15,
             'deactivatelightswashroom': 16,
             'increaseheatkitchen': 17,
             'increaseheatbedroom': 18,
             'increaseheatwashroom': 19,
             'decreaseheatkitchen': 20,
             'decreaseheatbedroom': 21,
             'decreaseheatwashroom': 22,
             'bringnewspapernone': 23,
             'bringjuicenone': 24,
             'bringsocksnone': 25,
             'change languageChinesenone': 26,
             'change languageKoreannone': 27,
             'change languageEnglishnone': 28,
             'change languageGermannone': 29,
             'bringshoesnone': 30
        }
