import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Gaussian
import numpy as np


class HMM(inference.SSM):
    def __init__(self):
        self.cpt = 0
        self.x = sample("x_0", Gaussian(0, 1))

    def step(self, y):
        self.cpt += 1
        self.x = sample(f"x_{self.cpt}", Gaussian(self.x, 2))
        observe(f"o_{self.cpt}", Gaussian(self.x, 0.5), y)
        return self.x


def model(data):
    hmm = HMM()
    for y in data:
        hmm.step(y)
    return hmm.x


with inference.ImportanceSampling(num_particles=1000):
    dist = infer(model, np.arange(20))
    print(f"Importance Sampling: {dist.stats()}")

with inference.MCMC(num_samples=1000):
    dist = infer(model, np.arange(20))
    print(f"MCMC: {dist.stats()}")

from typing_extensions import reveal_type

with inference.SMC(num_particles=1000) as infer:
    dists = infer.infer_stream(HMM, np.arange(20))  # type: ignore
    for d in dists:
        pass
    print(f"Particle Filter {d.stats()}")
