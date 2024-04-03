import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Bernoulli, Gaussian
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


with inference.ImportanceSampling(num_particles=1000):
    dist: Categorical[float] = infer(weird)  # type: ignore
    dist.plot()
    plt.show()
