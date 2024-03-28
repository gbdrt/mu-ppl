import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Empirical, Uniform, Bernoulli
import matplotlib.pyplot as plt


def plinko(n:int) -> float:
    if (n==0): 
        return 0
    else:
        x = sample("x", Bernoulli(0.5))
        return x + plinko(n-1)

with inference.BasicSampling(num_samples=1000):
    dist = infer(plinko, 100)
    plt.hist(dist.samples, range=(0, 100), bins=100, color='b')
    plt.show()
    # print(dist.samples)