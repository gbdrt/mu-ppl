from typing import Iterable, List

import mu_ppl.inference as inference
from mu_ppl import infer, sample, observe
from mu_ppl.distributions import Gaussian, split
import numpy as np


class HMM(inference.SSM):
    def __init__(self):
        self.cpt = 0
        self.x = sample("x_0", Gaussian(0, 1))

    def step(self, y: float) -> float:
        self.cpt += 1
        self.x = sample(f"x_{self.cpt}", Gaussian(self.x, 2))
        observe(f"o_{self.cpt}", Gaussian(self.x, 0.5), y)
        return self.x


def model(data: Iterable[float]) -> List[float]:
    hmm = HMM()
    res = []
    for y in data:
        res.append(hmm.step(y))
    return res


data = np.arange(20)

with inference.ImportanceSampling(num_particles=1000):
    dist = infer(model, data)
    means = np.array([d.stats()[0] for d in split(dist)])
    mse = np.sum((means - data) ** 2) / len(data)
    print(f"Importance Sampling: {mse}")

with inference.MCMC(num_samples=1000, warmups=1000):
    dist = infer(model, data)
    means = np.array([d.stats()[0] for d in split(dist)])
    mse = np.sum((means - data) ** 2) / len(data)
    print(f"MCMC: {mse}")


with inference.SMC(num_particles=1000) as infer:
    dists = infer.infer_stream(HMM, data)  # type: ignore
    mse = 0
    for i, d in enumerate(dists):
        mse += (d.stats()[0] - data[i]) ** 2
    mse = mse / len(data)
    print(f"Particle Filter {mse}")
