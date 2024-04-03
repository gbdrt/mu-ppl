import mu_ppl.inference as inference
from mu_ppl import *
import matplotlib.pyplot as plt


def weird():
    b = sample(Bernoulli(0.5))
    mu = 0.5 if (b == 1) else 1.0
    theta = sample(Gaussian(mu, 1.0))
    if theta > 0.0:
        observe(Gaussian(mu, 0.5), theta)
        return theta
    else:
        return weird()


with ImportanceSampling(num_particles=1000):
    dist: Categorical[float] = infer(weird)  # type: ignore
    viz(dist)
    plt.show()
