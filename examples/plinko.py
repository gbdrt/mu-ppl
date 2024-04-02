import numpy as np
import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Empirical, Uniform, Bernoulli
import matplotlib.pyplot as plt
import seaborn as sns


def plinko(n: int) -> float:
    if n == 0:
        return 0
    else:
        x = sample(Bernoulli(0.5))
        return x + plinko(n - 1)


res = [plinko(100) for _ in range(1000)]
sns.histplot(res, kde=True, stat="probability")
plt.show()
