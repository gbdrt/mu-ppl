from typing import Iterable, List
from mu_ppl import *
import numpy as np


class HMM(SSM):
    def __init__(self):
        self.cpt = 0
        self.x = sample(Gaussian(0, 1), name="x_0")

    def step(self, y: float) -> float:
        self.cpt += 1
        self.x = sample(Gaussian(self.x, 2), name=f"x_{self.cpt}")
        observe(Gaussian(self.x, 0.5), y, name=f"o_{self.cpt}")
        return self.x


def model(data: Iterable[float]) -> List[float]:
    hmm = HMM()
    res = []
    for y in data:
        res.append(hmm.step(y))
    return res


data = np.arange(20)

with ImportanceSampling(num_particles=1000):
    dist = infer(model, data)
    means = np.array([d.stats()[0] for d in split(dist)])
    mse = np.sum((means - data) ** 2) / len(data)
    print(f"Importance Sampling: {mse}")

with SimpleMetropolis(num_samples=1000, warmups=1000):
    dist = infer(model, data)
    means = np.array([d.stats()[0] for d in split(dist)])
    mse = np.sum((means - data) ** 2) / len(data)
    print(f"MCMC: {mse}")


with SMC(num_particles=1000) as smc:
    dists = smc.infer_stream(HMM, data)  # type: ignore
    mse = 0
    for i, d in enumerate(dists):
        mse += (d.stats()[0] - data[i]) ** 2
    mse = mse / len(data)
    print(f"Particle Filter {mse}")
