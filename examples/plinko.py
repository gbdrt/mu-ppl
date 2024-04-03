import numpy as np
from mu_ppl import sample
from mu_ppl.distributions import Bernoulli
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore


def plinko(n: int) -> float:
    if n == 0:
        return 0
    else:
        x = sample(Bernoulli(0.5))
        return x + plinko(n - 1)


res = [plinko(100) for _ in range(1000)]
sns.histplot(res, kde=True, stat="probability")
plt.show()
