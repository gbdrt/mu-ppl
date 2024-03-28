import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume, observe
from mu_ppl.distributions import Uniform, Binomial
import matplotlib.pyplot as plt


b0 = 493472
f0 = 241945


def laplace():
    p = sample("p", Uniform(0, 1))
    observe("obs", Binomial(b0, p), f0)
    return p


with inference.ImportanceSampling(num_particles=1000):
    dist = infer(laplace)
    print(dist.stats())
    dist.plot()
    plt.show()
