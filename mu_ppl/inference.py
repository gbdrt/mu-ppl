import numpy as np
from .distributions import Discrete, Empirical

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
        return dist.sample() # Draw sample

    def observe(self, dist, value):
        pass # Ignore


class Reject(Exception):
    pass


class RejectionSampling(Handler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def sample(self, dist):
        return dist.sample() # Draw sample

    def observe(self, dist, v):
        x = dist.sample() # Draw a sample
        if x != v:
            raise Reject # Reject if it's not the observation

    def infer(self, model, *args, **kwargs):
        def gen(): # Generate one sample
            while True:
                try:
                    return model(*args, **kwargs)
                except Reject:
                    pass

        return Empirical([gen() for _ in range(self.num_samples)])


class ImportanceSampling(Handler):
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.id = None
        self.scores = np.zeros(num_particles)

    def sample(self, dist):
        return dist.sample() # Draw sample

    def observe(self, dist, v):
        self.scores[self.id] += dist.log_prob(v) # Update the score

    def infer(self, model, *args, **kwargs):
        values = np.empty(self.num_particles)
        for i in range(self.num_particles): # Run num_particles executions
            self.id = i
            values[i] = model(*args, **kwargs)
        return Discrete(values, self.scores)


class MCMC(Handler):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.tape = []
        self.prob = {"score": 0, "trace": []}

    def sample(self, dist):
        v = self.tape.pop(0) if self.tape else dist.sample() # Reuse if possible otherwise draw a sample
        self.prob["trace"].append(v) # Add to the trace
        return v

    def observe(self, dist, v):
        self.prob["score"] += dist.log_prob(v) # Update the score

    def infer(self, model, *args, **kwargs):
        def _mh(old_prob, new_prob): # MH acceptance prob
            fw = -np.log(len(old_prob["trace"]))
            bw = -np.log(len(new_prob["trace"]))
            return min(1.0, np.exp(new_prob["score"] - old_prob["score"] + bw - fw))

        samples = []
        new_value = model(*args, **kwargs)  # Generate first trace

        for _ in range(self.num_samples):
            samples.append(new_value)  # Store current sample
            old_prob = self.prob  # Store current trace
            old_value = new_value  # Store current value

            regen_from = np.random.randint(len(self.prob["trace"]))
            self.tape = self.prob["trace"][:regen_from]
            self.prob = {"score": 0, "trace": []}
            new_value = model(*args, **kwargs)  # Regen a new trace from regen_from

            if np.random.random() < _mh(old_prob, self.prob):  
                samples.append(new_value) # Keep the new trace and the new value
            else:
                new_value = old_value # Roll back to the previous value
                self.prob = old_prob # Restore previous trace

        return Empirical(samples)

# def SMC(Handler):
#     def __init__(num_particles):
#         self.num_particles = num_particles
#         self.prob = 