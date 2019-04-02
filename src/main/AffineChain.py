

import torch

def scalar_affine_chain(multipliers, summands):
    in_length = multipliers.shape[2]
    if in_length == 1:
        return (multipliers, summands)
    split = in_length // 2
    f_m, f_s = scalar_affine_chain(multipliers[:,:,:split], summands[:,:,:split])
    s_m, s_s = scalar_affine_chain(multipliers[:,:,split:], summands[:,:, split:])

    s_s += f_s[:,:,-1].unsqueeze(-1).expand_as(s_s) * s_m
    s_m *= f_m[:,:,-1].unsqueeze(-1)

    return torch.cat((f_m, s_m), dim=2), torch.cat((f_s, s_s), dim=2)


mults = 0.9*torch.ones(2,2,10)
sums = torch.ones(2,2,10)

print(scalar_affine_chain(mults, sums))