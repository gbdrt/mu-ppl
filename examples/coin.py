import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Uniform, Bernoulli


def coin(obs):
    p = sample("p", Uniform(0, 1))
    for i, o in enumerate(obs):
        observe(f"o_{i}", Bernoulli(p), o)
    return p


print(coin([0, 1]))

with inference.RejectionSampling(num_samples=10):
    dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist.stats())

with inference.ImportanceSampling(num_particles=1000):
    dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist.stats())

with inference.MCMC(num_samples=1000):
    dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist.stats())
