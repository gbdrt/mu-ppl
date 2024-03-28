import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume, observe
from mu_ppl.distributions import Uniform, Binomial, Bernoulli
import matplotlib.pyplot as plt

def answer (p):
    smoke = sample("s", Bernoulli(p))
    coin = sample( "c", Bernoulli(0.5))
    return coin or smoke

def canabis_hard():
    p = sample("u", Uniform(0, 1))
    yeses = np.sum([answer(p) for _ in range(200)])
    assume(yeses == 160)
    return p

# with inference.RejectionSampling(num_samples=1000):
#     dist = infer(canabis_hard)
#     dist.hist()
#     plt.show()

def canabis_soft():
    p = sample("u", Uniform(0, 1))
    yes_prop = np.mean([answer(p) for _ in range(200)])
    observe("bin", Binomial(200, yes_prop), 160)
    return p

with inference.ImportanceSampling(num_particles=1000):
    dist = infer(canabis_soft)
    dist.plot()
    plt.show()
    


def canabis_yes():
    smoke = sample("s", Bernoulli(p))
    coin = sample( "c", Bernoulli(0.5))
    assume(coin or smoke)
    return smoke

with inference.RejectionSampling(num_samples=1000):
    dist = infer(canabis_yes)
    dist.hist()
    plt.show()
    print(dist.stats())

