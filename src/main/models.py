
import torch
import torch.nn as nn
import torch.nn.functional as F

from ConvLM import CausalConv1d
from operator import mul

class ResBlock(nn.Module):
    def __init__(self, base):
        super(ResBlock, self).__init__()
        self.base = base

    def forward(self, x):
        return x + self.base(x)

def grouped_seq_res_ff(chan, hidden, activation, groups):
    return ResBlock(nn.Sequential(nn.Conv1d(chan, hidden, 1, groups=groups),
                                  activation,
                                  nn.Conv1d(hidden, chan, 1, groups=groups)))

def swish(x, b):
    return x * torch.sigmoid(b * x)

class Functional(nn.Module):
    def __init__(self, f):
        super(Functional, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class Swish(nn.Module):
    def __init__(self, chan):
        super(Swish, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(chan)))

    def forward(self, x):
        #print(x.shape)
        return swish(x, self.weight)


from functools import reduce
from hypercuboid import gen_hypercuboid

class ApplyConv(nn.Module):
    def __init__(self, func):
        super(ApplyConv, self).__init__()
        self.func = func

    def forward(self, x):
        x = x.permute(0, 2, 1)
        shape = x.shape
        return self.func(x.permute(0, 2, 1).reshape(-1, shape[2])).reshape(shape).permute(0, 2, 1)


def res_conv_lm(layers, channels, p_builder, kernel):
    spatials = [CausalConv1d(channels, channels, kernel, groups=channels) for _ in range(layers)]
    pointwise = [ResBlock(p_builder(i)) for i in range(layers)]
    layers = [nn.Embedding(256, channels), Functional(lambda x: x.permute(0, 2, 1))] + [x for pair in zip(spatials, pointwise) for x in pair]
    layers.append(nn.Conv1d(channels, 256, 1))
    return nn.Sequential(*layers)

def res_conv_sep_lm(layers, channels, p_builder, kernel):
    conv_builder = lambda x: ApplyConv(p_builder(x))
    return res_conv_lm(layers, channels, p_builder, kernel)

import SMoE
from normalization import ILBN
def smoe_conv_lm(layers, channels, n_experts=1024, ksize=8, hfactor=1, act=Swish(1), eps=0.1, gEps=0.1):
    experts = [nn.Sequential(ILBN(nn.Linear(channels, hfactor * channels)),
                             act,
                             nn.Linear(hfactor * channels, channels)) for _ in range(n_experts)]
    def p_builder(i):
        if i == layers - 1:
            return SMoE.SMoE(experts, channels, channels, eps, gEps)
        return nn.Sequential(ILBN(nn.Linear(channels, 4*channels)), Swish(1), nn.Linear(4*channels, channels))

    return res_conv_sep_lm(layers, channels, p_builder, ksize)

def standard_conv_lm(layers, channels, ksize=8, hfac=4, activation=Swish(1)):
    def builder(i):
        return nn.Sequential(nn.Conv1d(channels, channels*hfac, 1),
                                  activation,
                                  nn.Conv1d(channels*hfac, channels, 1))
    return res_conv_lm(layers, channels, builder, ksize)
    



                         
