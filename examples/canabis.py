import numpy as np
from mu_ppl import *
import matplotlib.pyplot as plt


def survey(p):
    smoke = sample(Bernoulli(p))
    coin = sample(Bernoulli(0.5))
    return coin or smoke


def canabis(yes, total):
    p = sample(Uniform(0, 1))
    smokers = np.sum([survey(p) for _ in range(total)])
    assume(yes == smokers)
    return p


with RejectionSampling(num_samples=100):
    dist: Empirical[float] = infer(canabis, 160, 200)  # type: ignore
    print(dist.stats())


def soldier():
    p = sample(dist)
    smoke = sample(Bernoulli(p))
    coin = sample(Bernoulli(0.5))
    assume(coin or smoke)
    return smoke


with RejectionSampling(num_samples=1000):
    dist: Empirical[float] = infer(soldier)  # type: ignore
    viz(dist)
    plt.show()
    print(dist.stats())
