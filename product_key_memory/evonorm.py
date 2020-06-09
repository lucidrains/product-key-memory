import torch
import torch.nn as nn
import torch.nn.functional as F

class SwishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    
class Swish(nn.Module):
    def forward(self, input_tensor):
        return SwishFn.apply(input_tensor)

# calculating cumulative variance
# expects tensor in the shape of (b, t, d), where 't' is the dimension to be cumulative
def cum_var(x):
    shape, device = x.shape, x.device
    b, t, d = shape
    x = x.reshape(b, t * d)

    x_cum, x2_cum = (x.cumsum(dim=1), (x ** 2).cumsum(dim=1))
    denom = torch.arange(0, t * d, device = device)[None, :] + 1
    cum_var = ((x2_cum - ((x_cum ** 2) / denom)) / denom).reshape(*shape)
    return cum_var[:, :, -1].unsqueeze(-1)

def group_std(x, groups = 32, causal = False, eps = 1e-5):
    shape = x.shape
    b, t, d = shape
    x = x.reshape(b, t, groups, d // groups)

    if causal:
        x_t = x.transpose(1, 2).reshape(b * groups, t, -1)
        var = cum_var(x_t).reshape(b, groups, t, -1).transpose(1, 2)
    else:
        var = torch.var(x, dim = (1, 3), keepdim = True)

    var = var.expand_as(x)
    return torch.sqrt(var + eps).reshape(*shape)

class EvoNorm1D(nn.Module):
    def __init__(self, dim, non_linear = True, eps = 1e-5, groups = 32, causal = False):
        super().__init__()
        assert (dim % groups == 0), f'dimension {dim} must be divisible by the number of groups {groups}'
        self.non_linear = non_linear
        self.swish = Swish()

        self.groups = groups
        self.eps = eps
        self.causal = causal

        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, **kwargs):
        if self.non_linear:
            num = self.swish(x)
            return num / group_std(x, groups = self.groups, causal = self.causal, eps = self.eps) * self.gamma + self.beta
        return x * self.gamma + self.beta
