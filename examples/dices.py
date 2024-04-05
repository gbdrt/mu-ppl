from mu_ppl import *
import matplotlib.pyplot as plt


def dice() -> int:
    a = sample(RandInt(1, 6), name="a")
    b = sample(RandInt(1, 6), name="b")
    return a + b


with Enumeration():
    dist: Categorical[float] = infer(dice)  # type: ignore
    print(dist.stats())
    viz(dist)
    plt.show()
