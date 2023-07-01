import json

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

from product_key_memory.product_key_memory import PKM

# helper function

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# sampling helpers

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, ind, val)
    return probs

# feedforward

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = heads * dim_head

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):
        n, h, device = x.shape[1], self.heads, x.device

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', h = h, qkv = 3)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        seq_len,
        pkm_layers = None,
        dim_head = 64,
        heads = 8,
        pad_value = 0,
        pkm_kwargs: dict = dict()
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pad_value = pad_value

        pkm_layers = default(pkm_layers, depth // 2)
        pkm_layers = (pkm_layers,) if not isinstance(pkm_layers, tuple) else pkm_layers
        pkm_layers = set(pkm_layers)

        if len(pkm_layers) > 0:
            print(f'using PKM at layers {pkm_layers}')
            print(json.dumps(pkm_kwargs, indent = 2))
            print('\n\n')

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.layers = nn.ModuleList([])

        for ind in range(depth):
            layer = ind + 1
            use_pkm = layer in pkm_layers

            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads),
                FeedForward(dim) if not use_pkm else PKM(dim, **pkm_kwargs)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompt,
        seq_len,
        temperature = 1.0,
        filter_thres = 0.9
    ):
        b, n, device = *prompt.shape, prompt.device

        out = prompt

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.seq_len:], return_loss = False)
            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)

            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim = -1)

        return out[:, n:]

    def forward(self, x, return_loss = True):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(x.shape[1], device = x.device))

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b c n -> b n c')

        return F.cross_entropy(logits, labels)
