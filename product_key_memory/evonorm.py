import torch
import torch.nn as nn

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

def group_std(x, groups = 32, eps = 1e-5):
    shape = x.shape
    b, t, d = shape
    x = x.reshape(b, t, groups, d // groups)
    var = torch.var(x, dim = (1, 3), keepdim = True).expand_as(x)
    return torch.sqrt(var + eps).reshape(*shape)

class EvoNorm1D(nn.Module):
    def __init__(self, dim, non_linear = True, eps = 1e-5, groups = 32):
        super().__init__()
        assert (dim % groups == 0), f'dimension {dim} must be divisible by the number of groups {groups}'
        self.non_linear = non_linear
        self.swish = Swish()

        self.groups = groups
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        if self.non_linear:
            num = self.swish(x)
            return num / group_std(x, groups = self.groups, eps = self.eps) * self.gamma + self.beta
        return x * self.gamma + self.beta
