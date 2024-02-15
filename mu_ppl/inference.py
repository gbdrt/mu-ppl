import numpy as np
from .distributions import Support

"""
Handler based inference à la Pyro
https://probprog.cc/2020/assets/posters/thu/49.pdf
"""

class Handler:
    def __enter__(self):
        global HANDLER
        self.old_handler = HANDLER
        HANDLER = self
        return self
    
    def __exit__(self, type, value, traceback):
        global HANDLER
        HANDLER = self.old_handler


HANDLER = Handler()

def sample(dist):
    return HANDLER.sample(dist)

def observe(dist, value):
    return HANDLER.observe(dist, value)

def infer(model, *args):
    return HANDLER.infer(model, *args)
    
    
class BasicSampler(Handler):
    def sample(self, dist):
        return dist.sample()
    
    def observe(self, dist, value):
        pass

class Reject(Exception):
    pass

class RejectionSampling(Handler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, dist):
        return dist.sample()

    def observe(self, dist, v):
        x = dist.sample()
        if x != v: raise Reject

    def infer(self, model, *args, **kwargs):
        def gen():
            while True:
                try:
                    return model(*args, **kwargs)
                except Reject: pass
        return [gen() for _ in range(self.num_samples)]


class ImportanceSampling(Handler):
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.id = None
        self.scores = np.zeros(num_particles)

    def sample(self, dist):
        return dist.sample()

    def observe(self, dist, v):
        self.scores[self.id] += dist.logpdf(v)

    def infer(self, model, *args, **kwargs):
        values = np.empty(self.num_particles)
        for i in range(self.num_particles):
            self.id = i
            values[i] = model(*args, **kwargs)
        return Support(values, self.scores)
    
class ImportanceSamplingVect(Handler):
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.scores = np.zeros(num_particles)

    def sample(self, dist):
        return dist.sample(size=self.num_particles)

    def observe(self, dist, v):
        self.scores += dist.logpdf(v)

    def infer(self, model, *args, **kwargs):
        values = model(*args, **kwargs)
        return Support(values, self.scores)



