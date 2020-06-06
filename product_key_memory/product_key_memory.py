import torch
from torch import nn

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

class PKM(nn.Module):
    def __init__(self, dim, heads = 8, num_keys = 128, topk = 10):
        super().__init__()
        assert (dim % heads == 0), 'dimension must be divisible by number of heads'
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        d_head = dim // heads
        self.to_queries = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.keys = nn.Parameter(torch.randn(heads, num_keys, 2, d_head))
        self.values = nn.Parameter(torch.randn(heads, num_keys ** 2, d_head))

    def forward(self, x):
        b, t, e, h = *x.shape, self.heads
        queries = self.to_queries(x).chunk(2, dim=-1)
        queries = torch.stack(queries).reshape(2, b, t, h, -1)

        dots = torch.einsum('pbthd,hnpd->bhtpn', queries, self.keys)
        scores, indices = dots.topk(k=self.topk, dim=-1)
        scores, indices = map(lambda x: x.chunk(2, dim=2), (scores, indices))

        shape = (b, h, t, self.topk ** 2)

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

        values = self.values[None, :, None, :, :].expand(b, -1, t, -1, -1)
        expanded_indices = expand_dim(value_indices, dim=4, k=values.shape[-1])
        selected_values = values.gather(-2, expanded_indices)

        out = (attn.unsqueeze(-1) * selected_values).sum(dim=-2)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)