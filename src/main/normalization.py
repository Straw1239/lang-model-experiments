import torch
import torch.nn as nn
import torch.nn.functional as F

def whitener(x, iters=5, p=None):
    dim = len(x)
    norm = torch.norm(x)
    x = x / norm
    if p is None:
        p = torch.eye(dim, device=x.device)
    for i in range(iters):
        p = 0.5 * (3*p - torch.mm(p, torch.mm(p, torch.mm(p, x))))
        #print(torch.norm(torch.mm(torch.mm(p, x), torch.t(p)) - torch.eye(dim, device=x.device)))
    return p / torch.sqrt(norm)

class ILBN(nn.Module):

    def __init__(self, lin, roll=0.9, iters=7, correct_grad=False):
        super(ILBN, self).__init__()
        inputs = lin.in_features
        self.register_buffer("linear", torch.eye(inputs, requires_grad=False))
        self.register_buffer("bias", torch.zeros(inputs, requires_grad=False))
        self.inputs = inputs
        self.lin = lin
        self.iters = iters
        self.register_buffer("eye", torch.eye(self.inputs))
        self.roll = roll
        self.eps = 1e-3;
        self.correct_grad = correct_grad

    def fused_apply(self, x):
        linear = self.linear
        bias = self.bias
        if not self.correct_grad:
            linear = linear.detach()
            bias = bias.detach()
        weight = torch.matmul(self.lin.weight, linear)
        bias = F.linear(bias, self.lin.weight, self.lin.bias)
        return F.linear(x, weight, bias)

    def update_whitener(self, x):
        grad_gate = torch.enable_grad() if self.correct_grad else torch.no_grad()

        with grad_gate:
            x = x.view(-1, self.inputs)
            sample_mean = torch.mean(x, 0)
            centered_x = x - sample_mean
            sample_cov = torch.matmul(torch.t(centered_x), centered_x) / x.shape[0]
            self.bias = self.bias.detach()
            self.linear = self.linear.detach()
            self.bias *= 0.9
            self.linear *= self.roll
            self.bias -= (0.1) * sample_mean
            self.linear += (1-self.roll) * whitener(sample_cov + self.eps * self.eye, self.iters)

    def forward(self, x):
        self.update_whitener(x)
        return self.fused_apply(x)
    
    
