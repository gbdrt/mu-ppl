import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Uniform, Bernoulli
import matplotlib.pyplot as plt


def coin(obs):
    p = sample(Uniform(0, 1), name="p")
    for i, o in enumerate(obs):
        observe(Bernoulli(p), o, name=f"o_{i}")
    return p


# print(coin([0, 1]))

# with inference.RejectionSampling(num_samples=10):
#     dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
#     print(dist.stats())
#     dist.hist()
#     plt.show()


# with inference.ImportanceSampling(num_particles=10000):
#     dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
#     print(dist.stats())
#     dist.plot()
#     plt.show()


with inference.MCMC(num_samples=1000):
    dist = infer(coin, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    print(dist.stats())
    dist.hist()
    plt.show()
