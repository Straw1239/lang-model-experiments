
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import pathlib
import os

import numpy as np
from autotune import LRTuner

def train(model, dataset, iterations, block_size, baseOpt=optim.SGD, lr=0.001, momentum=0.9, tune_ratio=1.2599, device=torch.device(0), tune_itrs=200):

    model = model.to(device)
    baseOpt = baseOpt(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    running_loss = torch.zeros(1, device=device)
    last_loss = torch.zeros(1, device=device)

    dataset = dataset.to(device)
    


    def step(lr):
        
        for i in range(tune_itrs):
            r = rand.randint(0, len(dataset) - block_size-1)
            seq = dataset[r : r+block_size+1]
            inputs = seq[:-1].unsqueeze(0)
            labels = seq[1:].unsqueeze(0)
            baseOpt.zero_grad()

            for g in baseOpt.param_groups:
                g['lr'] = lr
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            baseOpt.step()
            running_loss.add_(loss.detach())
            
            if i == 0:
                last_loss.copy_(loss.detach())
            result = last_loss - loss.detach()
        return result
        
    
    opt = LRTuner(lr, step, device, tune_ratio) 

    print_interval = 1
    nextPrint = print_interval
    for i in range(0, iterations, tune_itrs):
        opt.step()
        if i > nextPrint:
            print(running_loss.item() / print_interval, end=' ')
            print(opt.LRs[0].item())
            running_loss.zero_()
            nextPrint += print_interval

enwik8 = torch.from_numpy(np.fromfile('../../data/enwik8', dtype='uint8')).long()
#enwik8 = torch.randint(26,[5000],dtype=torch.long)

from models import res_conv_sep_lm
from models import shared_smoe_conv_lm

#train(res_hypercuboid_lm(3, [256], 8), enwik8, 1000000, 2000)
    
      
      

