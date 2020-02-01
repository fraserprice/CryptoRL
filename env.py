# Python imports
import math
import random
from enum import Enum

import numpy as np
import numdifftools as nd

from gym import Env, spaces
from market_data import MarketData, MarketImitation
from utils import Normalizer


class ObservationSpaces(Enum):
    HUNDRED_STEPS = 0
    THOUSAND_STEPS = 1
    MODEL_BASED = 2


class RewardModels(Enum):
    MONTE_CARLO = 0
    TEMPORAL_DIFFERENCE = 1


def generate_random(n_points, start, sd, invalid_prob=0.):
    points = [start]
    curr = start
    for _ in range(n_points - 1):
        curr = max(0, curr + random.gauss(0, sd))
        points.append(curr if random.random() > invalid_prob else -1)
    return points


# Action space is between 0 and 1
# {
#   buy_certainty,
#   sell_certainty,
#   hold_certainty,
#   amount
# }
#
# Observation space is
# {
#   cash_held,
#   quantity_held,
#   current_price,
#   total_worth,
#   <market_sate>
# }
class MarketEnv(Env):
    def __init__(self, market_data, starting_cash_max=10000, trade_fee=5, reward_model=RewardModels.MONTE_CARLO):
        obs_shape = 104
        self.action_space = spaces.Box(low=0, high=100, shape=(4,), dtype='float32')
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_shape,), dtype='float32')
        self.market_normalizer = Normalizer(obs_shape)
        self.state_normalizer = Normalizer(4)

        self.market_data = market_data
        self.reward_model = reward_model

        self.starting_cash_max = starting_cash_max
        self.starting_cash = random.uniform(0, starting_cash_max)
        self.cash = self.starting_cash
        self.current_price = 0
        self.quantity_held = 0
        self.net_assets = self.cash
        self.steps = 0
        self.episode_length = 200
        self.trade_fee = trade_fee

        self.reset()

    def step(self, action):
        self.steps += 1
        next = self.market_data.next()
        self.current_price = next if next != -1 else self.current_price

        buy, sell, hold, amount = [abs(a) for a in action]
        decision = max(buy, sell, hold)
        if buy == decision and self.cash >= self.current_price:
            amount = min(int(amount), int(self.cash / self.current_price) if self.current_price > 0 else 1000)
            cost = amount * self.current_price

            self.cash -= cost
            self.quantity_held += amount
        elif sell == decision and self.quantity_held >= 0:
            amount = min(int(amount), self.quantity_held)
            cost = amount * self.current_price

            self.cash += cost
            self.quantity_held -= amount

        obs = self.__get_observation()
        done = self.steps == self.episode_length

        if self.reward_model == RewardModels.MONTE_CARLO:
            reward = (0 if not done else self.net_assets - self.starting_cash)
        else:
            reward = self.net_assets - self.starting_cash

        return obs, reward, done, {}

    def reset(self):
        self.market_data.load_raw(generate_random(
            1000, max(0, random.gauss(20, 15)), max(0, random.gauss(1, 1)), invalid_prob=0.03
        ))
        self.market_data.current_point = random.randint(0, 100)

        self.starting_cash = random.uniform(0, self.starting_cash_max)
        self.current_price = self.market_data.data[self.market_data.current_point]
        self.cash = self.starting_cash
        self.net_assets = self.starting_cash
        self.quantity_held = 0
        self.steps = 0

        return self.__get_observation()

    def render(self, mode='human', close=False):
        import matplotlib.pyplot as plt

        print("Start: " + str(self.starting_cash))
        print("Net: " + str(self.net_assets))
        print("Cash: " + str(self.cash))
        print("Quantity: " + str(self.quantity_held))
        print("")

        return True

    def __get_observation(self):
        market_history = self.market_data.get_last_n_points(100)
        self.current_price = market_history[-1]
        self.net_assets = self.current_price * self.quantity_held + self.cash

        return np.array([self.cash, self.quantity_held, self.net_assets, self.current_price] + market_history)


class MarketOHLCVEnv(Env):
    def __init__(self, kaiko_data, starting_cash_max=10000, trade_fee=5, reward_model=RewardModels.MONTE_CARLO,
                 n_obs=100, ep_len=1000):
        self.n_obs = n_obs
        self.n_state = 4
        obs_shape= n_obs + self.n_state
        self.action_space = spaces.Box(low=0, high=100, shape=(4,), dtype='float32')
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_shape,), dtype='float32')
        self.OHLCV_normalizer = Normalizer(obs_shape)
        self.state_normalizer = Normalizer(self.n_state)

        self.kaiko_data = kaiko_data
        self.reward_model = reward_model

        self.starting_cash_max = starting_cash_max
        self.starting_cash = random.uniform(0, starting_cash_max)
        self.cash = self.starting_cash
        self.current_price = 0
        self.quantity_held = 0
        self.net_assets = self.cash
        self.steps = 0
        self.episode_length = ep_len
        self.trade_fee = trade_fee

        self.instrument = None
        self.OHLCV_window = None
        self.current_instrument = None
        self.normalized_window = None

        self.reset()

    # noinspection PyTypeChecker
    def step(self, action):
        self.steps += 1
        next_timestep = self.OHLCV_window[self.steps + self.n_obs]
        self.current_price = (next_timestep["high"] - next_timestep["low"]) / 2

        buy, buy_prop, sell, sell_prop, hold = [abs(a) for a in action]  # TODO: Short/Long
        decision = max(buy, sell, hold)
        if buy == decision and self.cash >= self.current_price:
            amount = buy_prop * (self.cash / self.current_price)
            cost = amount * self.current_price
            self.cash -= cost
            self.quantity_held += amount
        elif sell == decision and self.quantity_held >= 0:
            amount = sell_prop * self.quantity_held
            cost = amount * self.current_price
            self.cash += cost
            self.quantity_held -= amount

        return self.__get_observation(), self.__get_reward(), self.steps == self.episode_length, {}

    def reset(self):
        self.instrument, self.OHLCV_window = self.kaiko_data.get_random(n_points=self.episode_length + self.n_obs)
        self.starting_cash = random.uniform(0, self.starting_cash_max)
        self.cash = self.starting_cash
        self.net_assets = self.starting_cash
        self.quantity_held = 0
        self.steps = 0
        self.OHLCV_normalizer = Normalizer(self.n_obs)
        for obs_OHLCV in self.OHLCV_window[0:self.n_obs]:
            self.OHLCV_normalizer.observe(np.array(list(obs_OHLCV.values())))
        self.normalized_window = [self.OHLCV_normalizer.normalize(list(obs_OHLCV.values())) for obs_OHLCV in self.OHLCV_window[0:self.n_obs]]
        self.state_normalizer = Normalizer(self.n_state)

        return self.__get_observation()

    def __get_reward(self):
        return self.net_assets - self.starting_cash

    def render(self, mode='human', close=False):
        import matplotlib.pyplot as plt

        print("Start: " + str(self.starting_cash))
        print("Net: " + str(self.net_assets))
        print("Cash: " + str(self.cash))
        print("Quantity: " + str(self.quantity_held))
        print("")

        return True

    def __get_observation(self):
        del self.normalized_window[0]
        next_timestep = self.OHLCV_window[self.steps + self.n_obs]
        self.current_price = (next_timestep["high"] - next_timestep["low"]) / 2
        next_timestep = np.array(list(next_timestep.values()))
        self.OHLCV_normalizer.observe(next_timestep)
        self.normalized_window.append(self.OHLCV_normalizer.normalize(next_timestep))

        self.net_assets = self.current_price * self.quantity_held + self.cash
        state = np.array([
            self.cash,
            self.quantity_held,
            self.net_assets,
            self.current_price
        ])
        self.state_normalizer.observe(state)
        normalized_state = self.state_normalizer.normalize(state)

        return np.concatenate((normalized_state, self.normalized_window))

if __name__ == "__main__":
    print(generate_random(100, 10, 1))
