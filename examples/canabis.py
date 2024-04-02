import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume
from mu_ppl.distributions import Uniform, Bernoulli, Empirical
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


with inference.RejectionSampling(num_samples=100):
    dist: Empirical[float] = infer(canabis, 160, 200)  # type: ignore


def soldier():
    p = sample(dist)
    smoke = sample(Bernoulli(p))
    coin = sample(Bernoulli(0.5))
    assume(coin or smoke)
    return smoke


with inference.RejectionSampling(num_samples=1000):
    dist: Empirical[float] = infer(soldier)  # type: ignore
    dist.hist()
    plt.show()
    print(dist.stats())
