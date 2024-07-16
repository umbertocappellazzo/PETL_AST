#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:31:49 2023

@author: umbertocappellazzo
"""

import os
import librosa
from torch.utils.data import Dataset
import numpy as np
import soundfile
from transformers import AutoFeatureExtractor, AutoProcessor
import torchaudio, torch
import re
from pathlib import Path
import glob

class IEMOCAP(Dataset):
    
    """
    sessions: the session(s) we want to include. For training, usually 4 sessions are used, with both female and male speakers.
    speaker_id: this param is used when we consider the valid/test sets. For training, it must be set to 'both'. For the femal speaker, set it to 'F', otherwise 'M'.
    
    We consider only 4 labels: 0 --> neutral, 1 --> happy, 2 --> sad, 3 --> anger. Happy also includes the 'excitement' label.
    """
    
    def __init__(self, data_path, max_len_audio, max_len_AST, sessions, speaker_id, accept_labels = ['neu','hap','exc','sad','ang'], is_AST = False, apply_SpecAug= False, few_shot = False, samples_per_class = 1):
        
        self.max_len_audio = max_len_audio
        self.max_len_AST = max_len_AST
        self.data_path = os.path.expanduser(data_path)
        self.is_AST = is_AST
        self.sessions = sessions
        self.speaker_id = speaker_id
        self.accept_labels = accept_labels
        
        self.apply_SpecAug = apply_SpecAug
        self.freq_mask = 24
        self.time_mask = 140
        
        self.x, self.y = self.get_data()
        
        if few_shot:
            self.x, self.y = self.get_few_shot_data(samples_per_class)
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        if self.is_AST:
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
        x, y = [], []
        if self.is_AST:
            processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", max_length=self.max_len_AST)
        else:
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        
        for session in self.sessions:
            label_dir = os.path.join(self.data_path,'IEMOCAP_full_release',f'Session{session}/dialog/EmoEvaluation/*.txt')
            label_paths = glob.glob(label_dir)
            
            for label_path in label_paths:
                
                with open(label_path, "r") as f:
                    for line in f:
                        
                        if not line.startswith("["):
                            continue
                        line = re.split("[\t\n]", line)
                        wav_name = line[1]  # Ex.: wav_name = 'Ses01F_impro01_F000'
                        wav_name_split = wav_name.split('_')
                        if self.speaker_id != 'both': 
                            if self.speaker_id != wav_name_split[-1][0]:
                                continue
                        label = line[2]
                        
                        if label in self.accept_labels:
                            y.append(self.label_mapping[label])
                        else:
                            continue
                        
                        if len(wav_name_split) == 3:
                            pathh = os.path.join(self.data_path,'IEMOCAP_full_release',f'Session{session}/sentences/wav',wav_name_split[0]+'_'+wav_name_split[1],wav_name+'.wav')
                        else:
                            pathh = os.path.join(self.data_path,'IEMOCAP_full_release',f'Session{session}/sentences/wav',wav_name_split[0]+'_'+wav_name_split[1]+'_'+wav_name_split[2],wav_name+'.wav')
                            
                        wav,sampling_rate = soundfile.read(pathh)
                        if self.is_AST:
                            x.append(processor(wav, sampling_rate= 16000, return_tensors='pt')['input_values'].squeeze(0))
                            
                        else: 
                            x.append(processor(wav, padding = 'max_length', max_length=self.max_len_audio, truncation=True, return_tensors='pt', sampling_rate= 16000)['input_values'].squeeze(0))
        
        return np.array(x), np.array(y)
    
    @property
    def label_mapping(self):
        # 'hap' and 'exc' are typically grouped into the same emotion.
        
        return {'neu': 0,
                'hap': 1,
                'exc': 1,
                'sad': 2,
                'ang': 3,
            }