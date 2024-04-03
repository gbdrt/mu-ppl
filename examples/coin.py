from typing import List
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Uniform, Bernoulli, Empirical, Categorical
import matplotlib.pyplot as plt


def coin(obs: List[int]) -> float:
    p = sample(Uniform(0, 1), name="p")
    for i, o in enumerate(obs):
        observe(Bernoulli(p), o, name=f"o_{i}")
    return p


print(coin([0, 1]))

with inference.RejectionSampling(num_samples=10):
    dist1 = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist1.stats())
    dist1.hist()
    plt.show()


with inference.ImportanceSampling(num_particles=10000):
    dist2: Categorical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist2.stats())
    dist2.plot()
    plt.show()


with inference.MCMC(num_samples=2000, warmups=1000, thinning=2):
    dist3: Empirical[float] = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # type: ignore
    print(dist3.stats())
    dist3.hist()
    plt.show()
