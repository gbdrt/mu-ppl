from mu_ppl import *
import matplotlib.pyplot as plt


# def funny_bernoulli() -> int:
#     a = sample("a", Bernoulli(0.5))
#     b = sample("b", Bernoulli(0.5))
#     c = sample("c", Bernoulli(0.5))
#     return a + b + c


def funny_bernoulli() -> int:
    a = sample(Bernoulli(0.5), name="a")
    b = sample(Bernoulli(0.5), name="b")
    c = sample(Bernoulli(0.5), name="c")
    assume(a == 1 or b == 1)
    return a + b + c


# with RejectionSampling(num_samples=1000):
#     dist1: Empirical[int] = infer(funny_bernoulli)  # type: ignore
#     print(dist1.stats())
#     viz(dist1)
#     plt.show()

with Enumeration():
    dist2: Categorical[float] = infer(funny_bernoulli)  # type: ignore
    print(dist2.stats())
    viz(dist2)
    plt.show()
