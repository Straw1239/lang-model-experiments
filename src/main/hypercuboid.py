import torch.nn as nn

class HyperCuboid(nn.Module):
    def __init__(self, layers, dims):
        self.layers = layers
        self.dims = dims

    def forward(self, x):
        cuboid_shape = x.shape[:-1] + dims
        linear_shape = x.shape
        perm = [(i + 1) % len(self.dims) for i in range(len(self.dims))] 
        for layer in self.layers:
            x = layer(x)
            x = x.view(cuboid_shape).permute(perm).view(linear_shape).contigous()
        return x
        
from operator import mul
from functools import reduce
    
def gen_hypercuboid(dims, group_layer):
    prod = reduce(mul, dims, 1)
    return nn.Sequential(*[group_layer(prod // d) for d in dims])
    
    
    

