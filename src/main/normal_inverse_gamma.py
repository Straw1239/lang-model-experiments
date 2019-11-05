from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
import torch

class NIGDist():
    
    def __init__(self, m, vinv, a, b):
        self.m = m
        self.vinv = vinv
        self.a = a
        self.b = b

    def update(self, total, moment, n):
        
        newVinv = self.vinv  + n
        newM = (1.0 / newVinv) * (self.vinv * self.m + total)
        self.a += n / 2
        self.b += 0.5 * (self.m * self.m * self.vinv + moment - newM * newM * newVinv)
        self.m = newM
        self.vinv = newVinv
        
        
    def update(self, x):
        total = torch.sum(x, dim=1)
        moment = torch.sum(x * x, dim=1)
        n = x.shape[1]
        self.update(total, moment, n)

    def sample(self):
        vars = 1.0 / (Gamma(self.a, self.b)).sample()
        means = Normal(self.m, vars / self.vinv).sample()
        return means, vars

    def update_component(self, index, x):
        total = torch.sum(x)
        moment = torch.sum(x * x)
        n = x.shape[0]
        newVinv = self.vinv[index]  + n
        newM = (1.0 / newVinv) * (self.vinv[index] * self.m[index] + total)
        self.a[index] += n / 2
        self.b[index] += 0.5 * (self.m[index] * self.m[index] * self.vinv[index] + moment - newM * newM * newVinv)
        self.m[index] = newM
        self.vinv[index] = newVinv
        
        

def standardNIG(batch, device):
    return NIGDist(torch.zeros(batch, device=device), torch.ones(batch, device=device), torch.ones(batch, device=device), torch.ones(batch, device=device))

