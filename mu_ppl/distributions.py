import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
from abc import ABC


class Distribution(ABC):
    def sample(self, *args, **kwargs):
        return self.rv.rvs(*args, **kwargs)

    def logpdf(self, x, *args, **kwargs):
        match self.rv.dist:
            case stats.rv_continuous():
                return self.rv.logpdf(x, *args, **kwargs)
            case stats.rv_discrete():
                return self.rv.logpmf(x, *args, **kwargs)
            case _:
                raise RuntimeError


class Bernoulli(Distribution):
    def __init__(self, p):
        self.rv = stats.bernoulli(p)


class Support(Distribution):
    def __init__(self, values, logits):
        self.values = values
        self.logits = logits
        lse = logsumexp(logits)
        self.probs = np.exp(logits - lse)
        self.rv = stats.rv_discrete(values=(range(len(self.probs)), self.probs))

    def sample(self):
        return self.values[self.rv.rvs()]

    def logpdf(self, x):
        return self.rv.logpmf(self.values.index(x))
    
    def stats(self):
        mean = np.average(self.values, weights=self.probs)
        std = np.sqrt(np.cov(self.values, aweights=self.probs))
        return (mean, std)



class Uniform(Distribution):
    def __init__(self, a, b):
        self.rv = stats.uniform(a, b - a)


class Beta(Distribution):
    def __init__(self, a, b):
        self.rv = stats.beta(a, b)


class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        self.rv = stats.norm(loc=mu, scale=sigma)
