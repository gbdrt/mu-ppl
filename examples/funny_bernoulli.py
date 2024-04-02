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
    a = sample(Bernoulli(0.5), name="a")
    b = sample(Bernoulli(0.5), name="b")
    c = sample(Bernoulli(0.5), name="c")
    assume(a == 1 or b == 1)
    return a + b + c


with inference.RejectionSampling(num_samples=1000):
    dist = infer(funny_bernoulli)
    print(dist.stats())
    dist.hist()
    plt.show()

with inference.Enumeration():
    dist = infer(funny_bernoulli)
    print(dist.stats())
    dist.hist()
    plt.show()
