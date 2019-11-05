import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)

import models
    
class ConvLM(nn.Module):
    def __init__(self):
        super(ConvLM,self).__init__()
        self.embd = nn.Embedding(256, 256)
        self.conv1 = CausalConv1d(256, 256, 32, groups=64)
        
        self.conv2 = CausalConv1d(256, 256, 1)

    def forward(self, x):

        x = self.embd(x).permute(0,2,1)
        x = F.selu(self.conv1(x))
        
        x = self.conv2(x)
        return x
