import torch
import math
from torch import nn
from product_key_memory.evonorm import EvoNorm1D


def init_(t, dim = None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1. / math.sqrt(dim)
    return t.normal_(0, std)

def expand_dim(t, dim, k, unsqueeze = False):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def fetch_pkm_value_parameters(module):
    params = []
    for m in module.modules():
        if isinstance(m, PKM):
            params.append(m.values.weight)
    return params

class MergeDims(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.fn(x)
        return x.reshape(*shape)

class PKM(nn.Module):
    def __init__(self, dim, heads = 4, num_keys = 128, topk = 32, dim_head = 256, input_dropout = 0., query_dropout = 0., value_dropout = 0., use_evonorm = False):
        super().__init__()
        assert (dim % heads == 0), 'dimension must be divisible by number of heads'
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        dim_query = dim_head * heads
        self.to_queries = nn.Linear(dim, dim_query, bias = False)
        self.norm = MergeDims(nn.BatchNorm1d(dim_query)) if not use_evonorm else EvoNorm1D(dim_query)

        keys = init_(torch.randn(heads, num_keys, 2, dim_head // 2))
        self.keys = nn.Parameter(keys)
        self.values = nn.EmbeddingBag(num_keys ** 2, dim, mode='sum')

        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)
        self.value_dropout = nn.Dropout(value_dropout)

    def forward(self, x):
        b, t, e, h = *x.shape, self.heads
        x = self.input_dropout(x)

        queries = self.to_queries(x)
        queries = self.norm(queries)
        queries = self.query_dropout(queries)

        queries = queries.chunk(2, dim=-1)
        queries = torch.stack(queries).reshape(2, b, t, h, -1)

        dots = torch.einsum('pbthd,hnpd->bthpn', queries, self.keys)
        scores, indices = dots.topk(k=self.topk, dim=-1)
        scores, indices = map(lambda x: x.chunk(2, dim=3), (scores, indices))

        all_topk = self.topk ** 2
        shape = (b, t, h, all_topk)

        all_scores = (
            scores[0][..., :, None] +
            scores[1][..., None, :]
        ).reshape(*shape)

        all_indices = (
            indices[0][..., :, None] * self.num_keys +
            indices[1][..., None, :]
        ).reshape(*shape)

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        attn = final_topk.softmax(dim=-1)

        value_indices, attn = map(lambda x: x.reshape(-1, self.topk * h), (value_indices, attn))

        out = self.values(value_indices, per_sample_weights=attn)
        out = self.value_dropout(out)
        return out.reshape(b, t, e)

