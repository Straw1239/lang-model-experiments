import torch
import torch.distributions as D             
mean_zero = lambda x: torch.zeros(1)
import GP
from ringbuffer import RingBuffer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def uniform_grid(sizes, starts, ends):
    dim = len(sizes)
    result = torch.zeros(sizes + [dim])
    for i in range(dim):
        ruler = torch.linspace(starts[i], ends[i], sizes[i])
        slice_dim = result.select(dim, i)
        shape = [1 if j != i else sizes[i] for j in range(dim)]
        slice_dim += ruler.view(shape).expand_as(slice_dim)
        
    return result.view(-1, dim)

def normal_icdf(x):
    m = D.Normal(torch.zeros(1, device=x.device), torch.ones(1, device=x.device))
    return m.icdf(x)

def bmsqrt(A, numIters=8):
    dtype = A.dtype
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum().sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    
    return sA

def msqrt(x, numIters=8):
    return torch.squeeze(bmsqrt(torch.unsqueeze(x, 0), numIters))


def sample(mean, cov, shape=[]):
    transform = msqrt(cov)
    result = torch.randn(shape + [len(cov)])
    return F.linear(result, transform, mean)
    
def gaussian_grid(mean, cov, width):
    lower = torch.ones_like(mean) / (width + 1)
    upper = 1 - lower
    base = uniform_grid([width for _ in mean], lower, upper)
    base = normal_icdf(base)
    return mean + torch.matmul(base, msqrt(cov))
    

def sample_cov(x):
    return torch.matmul(torch.t(x), x) / len(x)
class GPTuner():

    def __init__(self, dim, stepper, kernel, obs_var, history_len=100):
        self.dim = dim
        self.stepper = stepper
        self.k = kernel
        self.history_x = RingBuffer([dim], history_len)
        self.history_y = RingBuffer([1], history_len)
        self.obs_var_p = obs_var
        self.eps = 0.00

    def obs_var(self):
        return F.softplus(self.obs_var_p)
        
    def predict(self, x):
        mean, var = GP.predict(self.history_x.all(),
                               self.history_y.all(),
                               self.obs_var().expand(len(self.history_y)),
                               x, mean_zero, self.k)
        var += self.eps * torch.eye(var.shape[0])
        return mean, var

    def thompson_step(self, grid):
        mean, var = self.predict(grid)
        choice = torch.argmax(sample(mean, var))
        print(grid[choice])
        result = self.stepper(grid[choice])
        self.history_x.append(grid[choice])
        self.history_y.append(result)
        return result

    def log_prob(self):
        locs = self.history_x.all()
        var = self.k(locs, locs) + self.obs_var() * torch.eye(len(locs)) 
        return D.MultivariateNormal(torch.zeros(len(locs)), var).log_prob(self.history_y.all())

    def sample_argmax(self, init_grid, samples=1024):
        mean, var = self.predict(init_grid)
        choices = torch.argmax(sample(mean, var, [samples]), dim=1)
        return init_grid[choices]
                              
import torch.optim as optim
def self_tuned_GP(step, dim, grid_width=8):
    cov_root = torch.eye(dim + 1, requires_grad=True)
    obsvar = torch.ones(1, requires_grad=True)
    k = GP.gaussian_kernel(cov_root)
    def no_time_step(x):
        return step(x[:-1]).detach()
    tuner = GPTuner(dim + 1, no_time_step, k, obsvar, 100)
    grid = gaussian_grid(torch.zeros(dim), torch.eye(dim), grid_width)
    grid = F.pad(grid, (0, 1))
        
    hyperopt = optim.SGD([cov_root, obsvar], lr=0.001)
    max_cov = torch.eye(dim)
    max_mean = torch.zeros(dim)
    def tuning_step():
        max_sample = tuner.sample_argmax(grid)[:, :dim]
        max_mean.mul_(0.9)
        max_mean.add_(0.1 * torch.mean(max_sample, 0))
        max_cov.mul_(0.9)
        max_cov.add_(0.1 * sample_cov(max_sample))
        
        grid[:, :dim].copy_(gaussian_grid(max_mean, max_cov, width=grid_width))
        tuner.thompson_step(grid)
        loss = -(tuner.log_prob().mean())
        print(max_mean, max_cov)
        loss.backward()
        hyperopt.step()
        grid[:, dim] += 1

    return tuning_step
        
    
    
        


            
import math


def logistic(x):
    "Numerically-stable logistic function."
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
    
def tensorized(f):
    def tfunc(x):
        return f(*x)
    return tfunc

def softplus(x):
    return math.log(1+math.exp(-abs(x))) + max(x,0)

def logistic_step(step, save, revert, n):

    def meta_step(logStartLR, decay, offset):
        decay = softplus(decay)
        decay /= n

        save()
        last = float('inf')
        for i in range(n):
            lr = math.exp(logStartLR) * logistic(-i * decay  + offset)
            loss = step(lr)
            s = float(loss) - last 
            if s > 2 or math.isnan(float(s)):
                revert()
                return
            last = loss

    return meta_step







            
        
    
    
    
