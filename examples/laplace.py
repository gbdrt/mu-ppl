from mu_ppl import *
import matplotlib.pyplot as plt


# # b0 = 493472
# # f0 = 241945

# b0 = 297018
# f0 = 145159


# def laplace() -> float:
#     p = sample(Uniform(0, 1))
#     observe(Binomial(b0, p), f0)
#     return p


fp, gp = 377555, 393386  # Paris    1745 - 1784
fl, gl = 698958, 737629  # Londres  1664 - 1758


def laplace(f1, g1, f2, g2) -> float:
    p = sample(Uniform(0, 1), name="p")
    q = sample(Uniform(0, 1), name="q")
    observe(Binomial(f1 + g1, p), g1, name="f1")
    observe(Binomial(f2 + g2, q), g2, name="f2")
    return q > p


# with ImportanceSampling(num_particles=100000):
#     dist: Categorical[float] = infer(laplace, fp, gp, fl, gl)  # type: ignore
#     # print(dist.stats())
#     viz(dist)
#     plt.show()

with MetropolisHastings(num_samples=1000, warmups=5000, thinning=2):
    dist: Empirical[float] = infer(laplace, fp, gp, fl, gl)  # type: ignore
    print(dist.stats())
    viz(dist)
    plt.show()
