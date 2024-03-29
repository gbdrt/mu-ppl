import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe, assume
from mu_ppl.distributions import Uniform, Bernoulli
import matplotlib.pyplot as plt


# def funny_bernoulli():
#     a = sample("a", Bernoulli(0.5))
#     b = sample("b", Bernoulli(0.5))
#     c = sample("c", Bernoulli(0.5))
#     return a + b + c


def funny_bernoulli():
    a = sample("a", Bernoulli(0.5))
    b = sample("b", Bernoulli(0.5))
    c = sample("c", Bernoulli(0.5))
    assume(a == 1 or b == 1)
    return a + b + c

    # a = sample("a", Bernoulli(0.5))
    # if a:
    #     b = sample("b", Bernoulli(0.5))
    #     return b
    # else:
    #     c = sample("c", Bernoulli(0.5))
    #     return c


# with inference.RejectionSampling(num_samples=10000):
#     dist = infer(funny_bernoulli)
#     print(dist.stats())
#     dist.hist()
#     plt.show()

with inference.Enumeration():
    dist = infer(funny_bernoulli)
    print(dist.stats())
    dist.plot()
    plt.show()
