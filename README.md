# µ-PPL

mu-PPL is a micro probabilistic programming language (PPL) on top of Python (mu is for the greek later µ).
Following recent PPLs ([WebPPL](http://webppl.org/), [Pyro](https://pyro.ai/), [Gen.jl](https://www.gen.dev/)) mu-PPL provides the following probabilistic operators:

- `sample` to draw a sample from a distribution
- `assume` to condition the model on a boolean condition (hard conditionning)
- `factor` / `observe` to update the score of the current execution (soft conditionning)

## Getting started

mu-PPL can be installed with:
```bash
$ pip install git+https://github.com/gbdrt/mu-ppl
```

Here is the mu-PPL program that computes the bias of coin from a series of observations.

```python
from typing import List
from mu_ppl import *

def coin(obs: List[int]) -> float:
    p = sample(Uniform(0, 1), name="p")
    for o in obs:
        observe(Bernoulli(p), o)
    return p

with ImportanceSampling(num_particles=10000):
    dist: Categorical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist.stats())
```

You should get the following output:

```bash
100%|███████████████████████████████████| 10000/10000 [00:02<00:00, 3785.50it/s]
(0.2511526566307631, 0.11951005324220002)
```

More examples are available in the `examples` directory.