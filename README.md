## Product Key Memory

[![PyPI version](https://badge.fury.io/py/product-key-memory.svg)](https://badge.fury.io/py/product-key-memory)

Standalone Product Key Memory module for augmenting Transformer models

## Install

```bash
$ pip install product-key-memory
```

## Usage

Replace the feedforwards in a Transformer with the following

```python
import torch
from product_key_memory import PKM

pkm = PKM(
    dim = 512,
    heads = 4,
    dim_head = 256,       # keep at 256 for best results
    num_keys = 256,       # number of subkeys, # values will be num_keys ^ 2
    topk = 32,            # the top number of subkeys to select
    use_evonorm = True    # usually PKM requires decent batch sizes with batchnorm to work well. this is an experimental feature using the new evonorm-s0 for batch-independent normalization
)

x = torch.randn(1, 1024, 512)
values = pkm(x) # (1, 1024, 512)
```

## Learning Rates

To give different learning rates to the value parameters of the product-key-memory network, use the following helper function.

```python
from torch.optim import Adam
from product_key_memory import fetch_pkm_value_parameters

# this helper function, for your root model, finds all the PKM models and the embedding bag weight parameters
pkm_parameters = fetch_pkm_value_parameters(model)

optim = Adam([
    {'params': model.parameters()},
    {'params': pkm_parameters, 'lr': 1e-2}
], lr=1e-3)
```

## Appreciation

Special thanks go to <a href="https://github.com/AranKomat">Aran</a> for encouraging me to look into this, and to <a href="https://github.com/madisonmay">Madison May</a> for his <a href="https://www.pragmatic.ml/large-memory-layers-with-product-keys/">educational blog post</a>, which helped me understand this better.

## Citations

```bibtex
@misc{lample2019large,
    title   = {Large Memory Layers with Product Keys},
    author  = {Guillaume Lample and Alexandre Sablayrolles and Marc'Aurelio Ranzato and Ludovic Denoyer and Hervé Jégou},
    year    = {2019},
    eprint  = {1907.05242},
    archivePrefix = {arXiv}
}
```
