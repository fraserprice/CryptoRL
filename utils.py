import numpy as np


class Normalizer:
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean[:]
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = self.mean_diff / self.n

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        return (inputs - self.mean) / obs_std


if __name__ == "__main__":
    n = Normalizer(2)
    n.observe(np.array([1, 2]))
    n.observe(np.array([0, 1]))
    n.observe(np.array([2, 3]))
    print(n.normalize(np.array([1, 2])))
