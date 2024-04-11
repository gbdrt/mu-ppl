from typing import Tuple
from mu_ppl import *
import matplotlib.pyplot as plt


def pi() -> bool:
    x = sample(Uniform(-1, 1))
    y = sample(Uniform(-1, 1))
    d = x**2 + y**2
    return d < 1


with RejectionSampling(num_samples=1000):
    dist: Empirical[float] = infer(pi)  # type: ignore
    viz(dist)
    plt.show()


def pi4(n: int) -> float:
    theta = sample(Uniform(0, 1))
    for _ in range(n):
        x = sample(Uniform(0, 1))
        y = sample(Uniform(0, 1))
        observe(Bernoulli(theta), x**2 + y**2 <= 1)
    return theta


with ImportanceSampling(num_particles=100):
    dist2: Categorical[float] = infer(pi4, 1000)  # type: ignore
    mean, std = dist2.stats()
    viz(dist2)
    plt.show()


# def disk() -> Tuple[float, float]:
#     x = sample(Uniform(-1, 1))
#     y = sample(Uniform(-1, 1))
#     d = x**2 + y**2
#     assume(d < 1)
#     return (x, y)


# with RejectionSampling(num_samples=1000):
#     dist: Empirical[Tuple[float, float]] = infer(disk)  # type: ignore
#     x, y = zip(*dist.samples)
#     sns.scatterplot(x=x, y=y)
#     plt.axis("scaled")
#     plt.show()
