import numpy as np
from mu_ppl import *
import matplotlib.pyplot as plt
from scipy.special import expit  # type: ignore

temps = np.array(
    [
        66,
        70,
        69,
        68,
        67,
        72,
        73,
        70,
        57,
        63,
        70,
        78,
        67,
        53,
        67,
        75,
        70,
        81,
        76,
        79,
        75,
        76,
        58,
    ]
)
fails = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1])


def logistic(a, b, t):
    return expit(-(a + b * t))


def challenger(temps, fails):
    a = sample(Gaussian(0, 10), name="a")
    b = sample(Gaussian(0, 10), name="b")
    for i, (t, f) in enumerate(zip(logistic(a, b, temps), fails)):
        observe(Bernoulli(t), f, name=f"o_{i}")
    return logistic(a, b, 31)


with MetropolisHastings(num_samples=2000, warmups=10000, thinning=2):
    dist: Empirical[float] = infer(challenger, temps, fails)  # type: ignore
    viz(dist)
    plt.show()
