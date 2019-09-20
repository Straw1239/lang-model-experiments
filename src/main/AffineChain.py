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



def batch_mm(a, b):
    mat_dim = a.shape[-1]
    return torch.bmm(a.view(-1, mat_dim, mat_dim), b.view(-1, mat_dim, mat_dim)).view(a.shape)

def interleave(a, b, dim=0):
    pass

def affine_chain(multipliers, summands):
    batch = multipliers.shape[0]
    in_length = multipliers.shape[1]
    if in_length == 1:
        return (multipliers, summands)
    mult_even =  mutipliers[:,::2, :, :] 
    mult_odd =  mutipliers[:,1::2, :, :]
    mat_dim = multipliers.shape[2]
    mult_comb = batch_mm(mult_even, mult_odd)


    trans_even = summands[:,::2, :, :] 
    trans_odd =  summands[:,1::2, :, :]
    trans_comb = batch_mm(mult_odd, trans_even) + trans_odd
    mult_skele, trans_skele = affine_chain(mult_comb, trans_comb)

    result_mult_even = batch_mm(mult_skele[:,:-1,:,:], mult_even[:,1:,:,:])
    result_mult_even = torch.cat((multipliers[:,0], result_mult_even), 1)
    result_mult = interleave(result_mult_even, mult_skele, 1)

def serial_affine_chain(multipliers, summands):
    result_mults = torch.zeros_like(multipliers)
    result_adds = torch.zeros_like(summands)

    batch = multipliers.shape[0]
    in_length = multipliers.shape[1]

    result_mults[0, 0] = multipliers[0, 0]
    result_adds[0, 0] = summands[0, 0]
    for i in range(1, in_length):
        result_mults[:,i] = batch_mm(result_mults[:, i-1], multipliers[:, i])
        result_adds[:, i] = batch_mm(multipliers[:, i], result_adds[:, i-1]) + summands[:, i]

    return result_mults, result_adds
    

    
    
    

    
    
    
    
    




