import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, assume, observe
from mu_ppl.distributions import Uniform, Bernoulli, Empirical
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore


def model():
    x = sample(Uniform(-1, 1))
    y = sample(Uniform(-1, 1))
    d = x**2 + y**2
    assume(d < 1)
    return (x, y)


with inference.RejectionSampling(num_samples=1000):
    dist: Empirical[float] = infer(model)  # type: ignore
    x, y = list(zip(*dist.samples))
    sns.scatterplot(x=x, y=y)
    plt.axis("scaled")
    plt.show()
    # print(dist.samples)
