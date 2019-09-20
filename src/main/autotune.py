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



class ThompsonTuner():
    def __init__(self, stepper, num_options, device):
        self.stepper = stepper
        self.arms = num_options        
        self.belief = standardNIG(self.arms, device)
        self.histogram = torch.zeros(self.arms, device=device)
        self.n = 0
        self.ngpu = torch.zeros(1, device=device)
        self.device = device

    def step(self):
        samples, _ = self.belief.sample()
        play = torch.argmax(samples)
        obs = self.stepper(play)
        self.belief.update_component(play, obs)
        self.histogram[play] += 1
        self.n += 1
        self.ngpu += 1
        if self.arms == 2:
            pass # Special analytic case
        if self.n >= 100:
            #print(torch.max(self.histogram) / self.ngpu)
            if (torch.max(self.histogram) / self.ngpu) > 0.80:
                choice = torch.argmax(self.histogram)
                self.stepper.choose(choice)
                self.ngpu.zero_()
                self.n = 0
                self.histogram.zero_()
                self.belief.m = self.belief.m[choice].repeat(self.arms)
                self.belief.vinv = self.belief.vinv[choice].repeat(self.arms)
                self.belief.a = self.belief.a[choice].repeat(self.arms)
                self.belief.b = self.belief.b[choice].repeat(self.arms)
                self.belief.vinv /= 2
                self.belief.a /= 2
                self.belief.b /= 2
                
                
                

        
class LRTuner():
    #Note: Stepper produces correlated outputs. Take two steps per tuning step to avoid? (Overlapping minibatches?)
    class Stepper():
        def __init__(self, base):
            self.base = base
            
        def __call__(self, choice):
            return self.base.opt(self.base.LRs[choice])
            

        def choose(self, choice):
            #print("chose!")
            ratio = self.base.ratio
            if choice == 0:
                self.base.LRs /= ratio
            else:
                self.base.LRs *= ratio
            
    def __init__(self, lr, opt, device, ratio=1.259921049):
        self.opt = opt
        self.ratio = ratio
        self.LRs = torch.tensor([lr, lr * ratio], device=device)
        self.tuner = ThompsonTuner(LRTuner.Stepper(self), 2, device)

    def step(self):
        self.tuner.step()

    def zero_grad(self):
        self.opt.zero_grad()
    
    
    
