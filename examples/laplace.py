from mu_ppl import *
import matplotlib.pyplot as plt


# b0 = 493472
# f0 = 241945

b0 = 297018
f0 = 145159


def laplace() -> float:
    p = sample(Uniform(0, 1))
    observe(Binomial(b0, p), f0)
    return p


with ImportanceSampling(num_particles=1000):
    dist: Categorical[float] = infer(laplace)  # type: ignore
    print(dist.stats())
    viz(dist)
    plt.show()
