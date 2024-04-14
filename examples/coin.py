from typing import List
from mu_ppl import *
import matplotlib.pyplot as plt


def coin(obs: List[int]) -> float:
    p = sample(Uniform(0, 1), name="p")
    for i, o in enumerate(obs):
        observe(Bernoulli(p), o, name=f"o_{i}")
    return p


with RejectionSampling(num_samples=100):
    dist1: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist1.stats())
    print(len(dist1.samples))
    # viz(dist1)
    # plt.show()


with ImportanceSampling(num_particles=10000):
    dist2: Categorical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist2.stats())
    # viz(dist2)
    # plt.show()

with SimpleMetropolis(num_samples=1000, warmups=10000, thinning=2):
    dist3: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist3.stats())
    # viz(dist3)
    # plt.show()


with MetropolisHastings(num_samples=2000, warmups=1000, thinning=2):
    dist4: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist4.stats())
    # viz(dist4)
    # plt.show()
