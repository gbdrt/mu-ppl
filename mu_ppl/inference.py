import numpy as np
from .distributions import Discrete, Empirical
from abc import ABC
from copy import deepcopy

"""
Handler based inference à la Pyro
https://probprog.cc/2020/assets/posters/thu/49.pdf
"""


class Handler:
    def __enter__(self):
        global _HANDLER
        self.old_handler = _HANDLER
        _HANDLER = self
        return self

    def __exit__(self, type, value, traceback):
        global _HANDLER
        _HANDLER = self.old_handler


_HANDLER = Handler()


def sample(dist):
    return _HANDLER.sample(dist)


def observe(dist, value):
    return _HANDLER.observe(dist, value)


def infer(model, *args):
    return _HANDLER.infer(model, *args)


class BasicSampler(Handler):
    def sample(self, dist):
        return dist.sample()  # Draw sample

    def observe(self, dist, value):
        pass  # Ignore


class Reject(Exception):
    pass


class RejectionSampling(Handler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, dist):
        return dist.sample()  # Draw sample

    def observe(self, dist, v):
        x = dist.sample()  # Draw a sample
        if x != v:
            raise Reject  # Reject if it's not the observation

    def infer(self, model, *args, **kwargs):
        def gen():  # Generate one sample
            while True:
                try:
                    return model(*args, **kwargs)
                except Reject:
                    pass

        return Empirical([gen() for _ in range(self.num_samples)])


class ImportanceSampling(Handler):
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.id = 0
        self.scores = np.zeros(num_particles)

    def sample(self, dist):
        return dist.sample()  # Draw sample

    def observe(self, dist, v):
        self.scores[self.id] += dist.log_prob(v)  # Update the score

    def infer(self, model, *args, **kwargs):
        values = [None for _ in range(self.num_particles)]
        for i in range(self.num_particles):  # Run num_particles executions
            self.id = i
            values[i] = model(*args, **kwargs)
        return Discrete(values, self.scores)


class MCMC(Handler):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.score = 0
        self.trace = []
        self.tape = []

    def sample(self, dist):
        if self.tape:  # Reuse if possible otherwise draw a sample
            v = self.tape.pop(0)
        else:
            v = dist.sample()
        self.trace.append(v)  # Add to the trace
        return v

    def observe(self, dist, v):
        self.score += dist.log_prob(v)  # Update the score

    def infer(self, model, *args, **kwargs):
        def _mh(old_trace, old_score, new_trace, new_score):  # MH acceptance prob
            fw = -np.log(len(old_trace))
            bw = -np.log(len(new_trace))
            return min(1.0, np.exp(new_score - old_score + bw - fw))

        samples = []
        new_value = model(*args, **kwargs)  # Generate first trace

        for _ in range(self.num_samples):
            samples.append(new_value)  # Store current sample
            old_score, old_trace = self.score, self.trace  # Store current state
            old_value = new_value  # Store current value

            regen_from = np.random.randint(len(self.trace))
            self.score = 0
            self.trace = []
            self.tape = self.trace[:regen_from]
            new_value = model(*args, **kwargs)  # Regen a new trace from regen_from

            if np.random.random() < _mh(old_trace, old_score, self.trace, self.score):
                samples.append(new_value)  # Keep the new trace and the new value
            else:
                new_value = old_value  # Roll back to the previous value
                self.score, self.trace = old_score, old_trace  # Restore previous state

        return Empirical(samples)


class SSM(ABC):
    def __init__(self):
        pass

    def step(self):
        pass


class SMC(ImportanceSampling):
    # Model must be expressed as a state machine (SSM).
    # Resample at each step
    def resample(self):
        d = Discrete(self.particles, self.scores)  
        self.particles = [ # Resample a new set of particles
            deepcopy(d.sample()) for _ in range(self.num_particles) 
        ]  
        self.scores = np.zeros(self.num_particles)  # Reset the score

    def infer(self, SSM: SSM, *args):
        self.particles = [SSM() for _ in range(self.num_particles)] # Initialise the particles
        for y in zip(*args): # At each step
            values = [None for _ in range(self.num_particles)]
            for i in range(self.num_particles):
                self.id = i
                values[i] = self.particles[i].step(*y) # Execute all the particles
            yield Discrete(values, self.scores) # Return current distribution
            self.resample() # Resample the particles
