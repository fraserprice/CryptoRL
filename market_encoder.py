from numpy.random import random


class MarketEncoder:
    def __init__(self):
        pass

    def encode(self, market_data):
        pass


class MarketImitation:
    def __init__(self):
        pass

    def generate_data(self):
        pass

    def generate_random(self, n_points, s_d, start, invalid_prob=0.):
        points = [start]
        curr = start
        for _ in range(n_points - 1):
            curr = max(0, curr + random.gauss(0, s_d))
            points.append(curr if random.random() > invalid_prob else -1)
        return points
