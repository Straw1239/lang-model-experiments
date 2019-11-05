import torch
import torch.distributions
import torch.nn.functional as F

def predict(data_x, data_y, y_var, x, mu, k):
    if len(data_x) == 0:
        return mu(x), k(x, x)
    x_data_covar = k(x, data_x)
    data_covar_inv = torch.inverse(k(data_x, data_x) + torch.diag(y_var))
    mean = mu(x) + torch.matmul(torch.matmul(x_data_covar, data_covar_inv), (data_y - mu(data_x)))
    var = k(x, x) - torch.matmul(torch.matmul(x_data_covar, data_covar_inv), x_data_covar.t())

    return mean.squeeze(), var






def inner_product(x, y, qform):
    return torch.matmul(torch.matmul(x, qform), torch.t(y))

def norm_squared(x, qform):
    vecs = x.shape[0]
    xt = torch.matmul(x, qform)
    return torch.bmm(xt.view(vecs, 1, -1), x.view(vecs, -1, 1)).flatten()

def gaussian_kernel(cov_root):
    def k(x, y):
        cov = torch.matmul(cov_root, torch.t(cov_root))
        xy = inner_product(x, y, cov)
        xx = norm_squared(x, cov)
        yy = norm_squared(y, cov)
        dist_square = xx.view(-1, 1).expand(-1, xy.shape[1]) - 2*xy + yy.view(1, -1).expand(xy.shape[0], -1)
        return torch.exp(-dist_square)
    return k




    


