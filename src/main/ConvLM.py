import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups):
        super(ShiftedConv1D).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation*(kernel_size-1), dilation=dilation, groups=groups)

    def apply(self, x):
        return self.conv(x)

class ConvLM(nn.Module):
    def __init__(self):
        super(ConvLM).__init__()
        self.conv1 = nn.Conv1d(10, 100, 3)
        self.conv2 = nn.Conv1d(110, 256, 3)


    def apply(self, x):
        l1out = F.selu(self.conv1(x))
        x = self.conv2(torch.cat(x, l1out))
        return x