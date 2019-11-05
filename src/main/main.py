
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import pathlib
import os

import numpy as np
import autotune
import GP
import SMoE
import torch.nn.functional as F


def tuning_opt(base_opt, itrs, tuner, overlap_batches=5):
    def opt(params, loader, lossf):
        base = [base_opt(params)]
        old_params = [torch.zeros_like(p) for p in params]
        reverted = [False]
        def step(lr):
            for p in base[0].param_groups:
                p['lr'] = lr
            base[0].zero_grad()
            loss = lossf(loader())
            print(loss, lr)
            loss.backward()
            base[0].step()
            return loss
            
        def save():
            for n, o in zip(params, old_params):
                o.copy_(n)
            reverted[0] = False
                
        def revert():
            for n, o in zip(params, old_params):
                n.detach().copy_(o)
            base[0] = base_opt(params)
            reverted[0] = True

        decay_step = autotune.tensorized(autotune.logistic_step(step, save, revert, itrs))
        def meta_step(x):
            batches = [loader() for _ in range(overlap_batches)]
            start_loss = torch.sum(torch.stack([lossf(b) for b in batches]))
            decay_step(x)
            if reverted[0]:
                return -torch.ones(1)
            end_loss = torch.sum(torch.stack([lossf(b) for b in batches]))
            return start_loss - end_loss

        tuner_inst = tuner(meta_step)

        def tuning_step():
            tuner_inst()
        return tuning_step
    return opt
        
def blockAutoLoader(dataset, block_size):
    def load():
        r = rand.randint(0, len(dataset) - block_size-1)
        seq = dataset[r : r+block_size+1]
        inputs = seq[:-1].unsqueeze(0)
        labels = seq[1:].unsqueeze(0)
        return inputs, labels
    return load

def autoRLMLoss(lm):
    def lossf(batch):
        inputs, labels = batch
        return F.cross_entropy(lm(inputs), labels)
    return lossf
            

def autor_train(lm, loader, optimizer, iterations):
    step = optimizer(list(lm.parameters()), loader, autoRLMLoss(lm))
    for i in range(0, iterations):
        step()



enwik8 = torch.from_numpy(np.fromfile('../../data/enwik8', dtype='uint8')).long()
#enwik8 = torch.randint(256,[100000000],dtype=torch.long)


def standard_train(lm, horizon=100, device=torch.device(0)):
    autor_train(lm.to(device),
                blockAutoLoader(enwik8.to(device), 8192),
                tuning_opt(
                    lambda x: optim.SGD(x, 0.001, 0.9),
                    horizon,
                    lambda s: autotune.self_tuned_GP(s, 3)), 1000)

import models
import ConvLM
from models import res_conv_sep_lm
from models import smoe_conv_lm
from models import *

#train(res_hypercuboid_lm(3, [256], 8), enwik8, 1000000, 2000)
    
      
      

