import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Gaussian
import numpy as np


class HMM(inference.SSM):
    def __init__(self):
        self.x = sample(Gaussian(0, 1))

    def step(self, y):
        self.x = sample(Gaussian(self.x, 2))
        observe(Gaussian(self.x, 0.5), y)
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

with inference.SMC(num_particles=1000):
    dists = infer(HMM, np.arange(20))
    for d in dists: pass
    print(f"Particle Filter {d.stats()}")
