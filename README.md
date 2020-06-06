# Product Key Memory

Standalone Product Key Memory module for augmenting Transformer models

## Usage

```python
import torch
from product_key_memory import PKM

pkm = PKM(
    dim = 512,
    heads = 8,
    num_keys = 512,       # number of subkeys, # values will be num_keys ^ 2
    topk = 10,            # the top number of subkeys to select
    share_kv = False      # share key/values across heads
)

x = torch.randn(1, 1024, 512)
values = pkm(x) # (1, 1024, 512)
```

## Appreciation

Special thanks go to <a href="https://github.com/madisonmay">Madison May</a> for his <a href="https://www.pragmatic.ml/large-memory-layers-with-product-keys/">educational blog post</a>, which helped me understand this better.

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
