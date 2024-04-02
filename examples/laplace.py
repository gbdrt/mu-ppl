import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume, observe
from mu_ppl.distributions import Uniform, Binomial, Categorical
import matplotlib.pyplot as plt


# b0 = 493472
# f0 = 241945

b0 = 297018
f0 = 145159


def laplace():
    p = sample(Uniform(0, 1))
    observe(Binomial(b0, p), f0)
    return p


with inference.ImportanceSampling(num_particles=1000):
    dist: Categorical[float] = infer(laplace)  # type: ignore
    print(dist.stats())
    dist.plot()
    plt.show()
