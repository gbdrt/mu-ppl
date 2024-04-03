from typing import List
from mu_ppl import *
import matplotlib.pyplot as plt


def coin(obs: List[int]) -> float:
    p = sample(Uniform(0, 1), name="p")
    for i, o in enumerate(obs):
        observe(Bernoulli(p), o, name=f"o_{i}")
    return p


print(coin([0, 1]))

with RejectionSampling(num_samples=10):
    dist1: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist1.stats())
    viz(dist1)
    plt.show()


with ImportanceSampling(num_particles=10000):
    dist2: Categorical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist2.stats())
    viz(dist2)
    plt.show()


with MCMC(num_samples=2000, warmups=1000, thinning=2):
    dist3: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist3.stats())
    viz(dist3)
    plt.show()
