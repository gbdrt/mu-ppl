import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe, assume
from mu_ppl.distributions import Uniform, Bernoulli, Gaussian
import matplotlib.pyplot as plt


def weird():
    b = sample("b", Bernoulli(0.5))
    mu = 0.5 if (b == 1) else 1.0
    theta = sample("theta", Gaussian(mu, 1.0))
    if theta > 0.0:
        observe("obs", Gaussian(mu, 0.5), theta)
        return theta
    else:
        return weird()


with inference.ImportanceSampling(num_particles=1000):
    dist = infer(weird)
    dist.plot()
    plt.show()
