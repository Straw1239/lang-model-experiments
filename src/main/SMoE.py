import torch
import torch.nn.functional as F
import torch.optim as optim
import itertools


import math

def sparse_MoE(experts, inputs, weightMat, numOut, epsilon=0.2, gradEps=0.2):
    topW, index = torch.sort(weightMat, 1, descending=True)
    cumW = torch.cumsum(topW, 1)
    device = inputs.device
    
    capped = torch.clamp(cumW, max=1-epsilon)
    sparse_weight = F.relu(capped - cumW + topW)
    
    softCost = torch.sum(torch.where(F.pad(sparse_weight[:, 1:], (0, 1), 'constant', 0) > 0, torch.ones(1, device=device), sparse_weight), 1)
        
    sparse_weight = torch.gather(sparse_weight, 1, index)
    
    
    usage = sparse_weight > 0;
    


    
    topG, Gindex = torch.sort(weightMat / torch.sum(weightMat, 0), 0, descending=True)
    GcumW = torch.cumsum(topG, 0)
    GSW =  (torch.clamp(F.pad(GcumW[1:, :], (0, 0, 0, 1), 'constant', 1), max=1 - gradEps) - GcumW) > 0
    
    usage |= torch.gather(GSW, 0, Gindex)
    
    
    outputs = torch.zeros(len(inputs), numOut, device=device)
    usage_t = usage.t().contiguous()
    for i, e in enumerate(experts):
        input_indexes = torch.nonzero(usage_t[i])
        einput = torch.index_select(inputs, 0, input_indexes.squeeze())
        #print(input_indexes.shape)
        outputs.scatter_add_(0, input_indexes.expand(-1, outputs.shape[1]), sparse_weight[input_indexes, i] * e(einput))
    return outputs, softCost


import torch.nn as nn
import models

def test_SMoE(nExp=8192):
    experts = [nn.Sequential(nn.Linear(128, 256), models.Swish(1), nn.Linear(256, 128)) for _ in range(nExp)]
    inputs = torch.randn(256, 128) * 5
    selector = nn.Sequential(nn.Linear(128, nExp), nn.Softmax(1))

    copt = optim.SGD(selector.parameters(), lr=0.001, momentum=0.9)
    for i in range(1000):
        
        weightMat = selector(inputs)

        out, cost = sparse_MoE(experts, inputs, weightMat, 128)
        copt.zero_grad()
        cost.sum().backward();
        copt.step()
        print(cost)
        
    print(out)

if __name__=='__main__':
    test_SMoE()

costHistory = []
from normalization import ILBN
class SMoE(nn.Module):
    def __init__(self, experts, nIn, nOut, epsilon=0.2, gEps=0.2, sFactor=5):
        
        super(SMoE, self).__init__()
        self.experts = experts
        for idx, e in enumerate(experts):
            self.add_module(str(idx), e)
        self.selector = nn.Sequential(ILBN(nn.Linear(nIn, len(experts))), nn.Softmax(1))
        self.selector[0].lin.weight.data *= sFactor
        self.selector[0].lin.bias.data *= sFactor
        self.nOut = nOut
        self.eps = epsilon
        self.gEps = gEps



    def forward(self, x):
        weightMat = self.selector(x)
        out, costs = sparse_MoE(self.experts, x, weightMat, self.nOut, self.eps, self.gEps)
        costHistory.append(costs)
        return out
                         

    

    
    
      
    
        


    
