import math
import torch
from torch import nn, einsum

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# init

def init_(t, dim = None):
    dim = default(dim, t.shape[-1])
    std = 1. / math.sqrt(dim)
    return nn.init.normal_(t, mean=0, std=std)

# optimizer

def list_subtract(l, r):
    return [el for el in l if el not in set(r)]

def fetch_pkm_value_parameters(module):
    params = []
    for m in module.modules():
        if isinstance(m, PKM):
            params.append(m.values.weight)
    rest = list_subtract(module.parameters(), params)
    return params, rest

def fetch_optimizer_parameters(module, pkm_learning_rate = 1e-2):
    pkm_params, rest = fetch_pkm_value_parameters(module)
    return [{'params': rest}, {'params': pkm_params, 'lr': pkm_learning_rate}]

# norm

class MaskedBatchNorm1D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(
        self,
        x,
        mask = None
    ):
        if exists(mask):
            initial_x = x
            x = x[mask]

        x = self.fn(x)

        if exists(mask):
            initial_x[mask] = x
            x = initial_x

        return x

class PKM(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        num_keys = 128,
        topk = 32,
        dim_head = 128,
        input_dropout = 0.,
        query_dropout = 0.,
        value_dropout = 0.,
        use_layernorm = False,
        pre_layernorm = False
    ):
        super().__init__()
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        dim_query = dim_head * heads * 2
        self.to_queries = nn.Linear(dim, dim_query, bias = False)

        # pre-layernorm pattern

        self.pre_layernorm = nn.LayerNorm(dim) if pre_layernorm else nn.Identity()

        # batchnorm would break causality

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(dim_head)
        else:
            self.norm = MaskedBatchNorm1D(nn.BatchNorm1d(dim_head))

        self.keys = nn.Parameter(torch.zeros(heads, num_keys, 2, dim_head))
        self.values = nn.EmbeddingBag(num_keys ** 2, dim, mode = 'sum')
        init_(self.keys)
        init_(self.values.weight)

        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)
        self.value_dropout = nn.Dropout(value_dropout)

    def forward(
        self,
        x,
        input_mask = None,
        **kwargs
    ):
        b, t, h = *x.shape[:2], self.heads

        x = self.pre_layernorm(x)
        x = self.input_dropout(x)

        queries = self.to_queries(x)

        # split out query heads

        queries = rearrange(queries, 'b t (p h d) -> (b p h) t d', p = 2, h = h)

        # norm and dropout queries

        norm_kwargs = dict(mask = input_mask) if not self.use_layernorm else dict()
        queries = self.norm(queries, **norm_kwargs)
        queries = self.query_dropout(queries)

        # ready queries

        queries = rearrange(queries, '(b p h) t d -> p b t h d', p = 2, h = h)

        # similarity to keys

        dots = einsum('p b t h d, h n p d -> b t h p n', queries, self.keys)

        # topk scores

        scores, indices = dots.topk(k = self.topk, dim = -1)

        (scores_x, scores_y), (indices_x, indices_y) = map(lambda t: t.chunk(2, dim = 3), (scores, indices))

        all_topk = self.topk ** 2

        all_scores = rearrange((
            rearrange(scores_x, '... k -> ... k 1') +
            rearrange(scores_y, '... k -> ... 1 k')
        ), 'b t h ... -> b t h (...)')

        all_indices = rearrange((
            rearrange(indices_x, '... k -> ... k 1') * self.num_keys +
            rearrange(indices_y, '... k -> ... 1 k')
        ), 'b t h ... -> b t h (...)')

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        # attention

        attn = final_topk.softmax(dim=-1)

        value_indices, attn = map(lambda t: rearrange(t, 'b t h k -> (b t) (h k)'), (value_indices, attn))

        # aggregate

        out = self.values(value_indices, per_sample_weights=attn)
        out = self.value_dropout(out)

        return rearrange(out, '(b t) d -> b t d', b = b)
