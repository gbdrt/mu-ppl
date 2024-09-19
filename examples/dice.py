from mu_ppl import *
import matplotlib.pyplot as plt


def sum_dice() -> int:
    a = sample(RandInt(1, 6), name="a")
    b = sample(RandInt(1, 6), name="b")
    return a + b


with Enumeration():
    dist: Categorical[float] = infer(sum_dice)  # type: ignore
    print(dist.stats())
    viz(dist)
    plt.show()

def hard_dice() -> int:
    a = sample(RandInt(1, 6), name="a")
    b = sample(RandInt(1, 6), name="b")
    assume(a!=b)
    return a + b


with Enumeration():
    dist2: Categorical[float] = infer(hard_dice)  # type: ignore
    print(dist2.stats())
    viz(dist2)
    plt.show()
