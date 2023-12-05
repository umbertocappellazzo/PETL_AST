#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:18:19 2023

@author: umbertocappellazzo
"""
import torch

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train(True)
    
    loss = 0.
    correct = 0
    total = 0
    
    for idx_batch, (x, y) in enumerate(loader):
        
        optimizer.zero_grad()
        
        x = x.to(device)
        y = y.to(device)
        
        outputs = model(x)
        
        loss_batch = criterion(outputs,y)     
        loss += loss_batch.detach().item()
        total += len(x)
        correct += (y==outputs.argmax(dim=-1)).sum().item()
        
        loss_batch.backward()
        optimizer.step()
        scheduler.step()
    
    loss /= len(loader)
    accuracy = correct/total
    
    return loss, accuracy


def eval_one_epoch(model, loader, device, criterion):
    
    loss = 0.
    correct = 0
    total = 0  
    
    model.eval()
    
    with torch.inference_mode():
        for idx_batch, (x,y) in enumerate(loader): 
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            
            loss_batch = criterion(outputs, y)
            loss += loss_batch.detach().item()
            
            total += len(x)
            correct += (y==outputs.argmax(dim=-1)).sum().item()
        
        loss /= len(loader)
        
        accuracy = correct/total
        
    return loss, accuracy