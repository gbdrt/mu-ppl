import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume, observe
from mu_ppl.distributions import Uniform, Binomial, Bernoulli, Gaussian
import matplotlib.pyplot as plt

temps = np.array([66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58])
fails = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1])

def logistic(a, b, t):
    return 1 / (1 + np.exp(( b * t + a )))

def challenger(temps, fails):
    a = sample("a", Gaussian(0, 31))
    b = sample("b", Gaussian(0, 31))
    [ observe("o", Bernoulli(q), f) for (q, f) in zip(logistic(a, b, temps), fails) ]
    return sample("crash", Bernoulli(logistic(a, b, 31)))

with inference.MCMC(num_samples=120000, warmups=100000, thin=2):
    dist = infer(challenger, temps, fails)
    dist.hist()
    plt.show()