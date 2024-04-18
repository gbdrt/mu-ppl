from mu_ppl import *
import numpy as np


def gauss(v, m0, s0, s1) -> float:
    m = sample(Gaussian(m0, s0), name="m")
    observe(Gaussian(m, s1), v)
    return m


def exact(v, m0, s0, s1):
    sq = 1 / (1 / (s0 * s0) + 1 / (s1 * s1))
    m1 = (m0 / (s0 * s0) + v / (s1 * s1)) * sq
    return (m1, np.sqrt(sq))


with RejectionSampling(num_samples=10000):
    m0 = 5
    s0 = 4
    s1 = 3
    v = 2
    dist1: Empirical[float] = infer(gauss, v, m0, s0, s1)  # type: ignore
    print(dist1.stats())
    print(exact(v, m0, s0, s1))
    # plt.show()
