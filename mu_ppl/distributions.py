import numpy as np
import numpy.random as rand
from abc import ABC


class Distribution(ABC):
    def sample(self, *args, **kwargs):
        pass

    def log_prob(self, x, *args, **kwargs):
        pass

    def stats(self):
        pass


class Bernoulli(Distribution):
    def __init__(self, p):
        self.p = p

    def sample(self, *args, **kwargs):
        return rand.binomial(1, self.p, *args, **kwargs)

    def log_prob(self, v):
        return v * np.log(self.p) + (1 - v) * np.log(1 - self.p)

    def stats(self):
        return (self.p, self.p * self.q)


class Uniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self, *args, **kwargs):
        return rand.uniform(self.a, self.b, *args, **kwargs)

    def log_prob(self):
        return 1.0

    def stats(self):
        return ((self.b - self.a) / 2, np.sqrt((self.b - self.a) ** 2 / 12))


class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, *args, **kwargs):
        return rand.normal(self.mu, self.sigma, *args, **kwargs)

    def log_prob(self, v):
        return (
            1
            / (self.sigma * np.sqrt(2 * np.pi))
            * np.exp(((v - self.mu) / self.sigma) ** 2 / 2)
        )

    def stats(self):
        return (self.mu, self.sigma)


class Discrete(Distribution):
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        lse = np.log(np.sum(np.exp(logits)))
        self.probs = np.exp(logits - lse)

    def sample(self, *args, **kwargs):
        i = rand.multinomial(1, self.probs, *args, **kwargs)
        return self.values[i]

    def log_prob(self, v):
        i = self.values.index(v)
        return np.log(self.probs[i])

    def stats(self):
        mean = np.average(self.values, weights=self.probs)
        std = np.sqrt(np.cov(self.values, aweights=self.probs))
        return (mean, std)


class Empirical(Distribution):
    def __init__(self, samples):
        self.samples = samples

    def sample(self):
        i = rand.randint(len(self.samples))
        return self.samples[i]

    def log_prob(self, v):
        return 1 / len(self.samples) if v in self.samples else 0

    def stats(self):
        return (np.mean(self.samples), np.std(self.samples))
