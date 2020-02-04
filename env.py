# Python imports
import random
import time
from collections import defaultdict
from enum import Enum

import numpy as np

from gym import Env, spaces

from market_data import KaikoData
from sklearn.preprocessing import StandardScaler


class ObservationSpaces(Enum):
    HUNDRED_STEPS = 0
    THOUSAND_STEPS = 1
    MODEL_BASED = 2


class RewardModels(Enum):
    MONTE_CARLO = 0
    TEMPORAL_DIFFERENCE = 1


class MarketOHLCVEnv(Env):
    def __init__(self, trade_fee=5, reward_model=RewardModels.MONTE_CARLO,
                 n_obs=50, ep_len=100):
        self.kaiko_data = KaikoData()
        self.kaiko_data.load(min_points=n_obs + ep_len)
        self.n_obs = n_obs
        self.n_state = 4
        self.ohlcv_size = 3
        obs_shape = n_obs * self.ohlcv_size + self.n_state
        self.action_space = spaces.MultiDiscrete([3])
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_shape,), dtype='float32')
        self.OHLCV_normalizer = StandardScaler()
        self.state_normalizer = StandardScaler()
        self.reward_normalizer = StandardScaler()

        self.reward_model = reward_model
        self.steps = 0
        self.episode_length = ep_len
        self.trade_fee = trade_fee
        self.done = False
        self.buy_timestep = None
        self.buy_value = None
        self.instrument = None
        self.OHLCV_window = None
        self.current_instrument = None
        self.normalized_window = np.array([])


        self.reset()

    # noinspection PyTypeChecker
    def step(self, action):
        timestep, open_p, _, _, _, _ = self.__get_ohlcv_index(self.steps + self.n_obs - 1)

        buy, sell, hold = [action == a for a in range(0, 3)]  # TODO: Short/Long
        reward = 0
        if open_p == 0 or hold:
            print("Holding")
        elif buy and self.buy_timestep is None:
            print(f"Buy at {timestep} for {open_p}")
            self.buy_timestep, self.buy_value = timestep, open_p
        elif sell and self.buy_timestep is not None:
            reward = (open_p - self.buy_value) / self.buy_value
            print(f"Selling. Reward: {reward}")
            self.done = True
        if self.steps == self.episode_length:
            print("Episode ending. Failed...")
            reward = -10
        self.steps += 1

        return self.__get_observation(), reward, self.steps == self.episode_length or self.done, {}

    def reset(self):
        self.done = False
        self.buy_timestep, self.buy_value = None, None
        self.instrument, self.OHLCV_window = self.kaiko_data.get_random(n_points=self.episode_length + self.n_obs)
        self.steps = 0
        self.OHLCV_normalizer = StandardScaler()
        self.normalized_window = self.OHLCV_normalizer.fit_transform(self.__get_ohlcv_range(0, self.n_obs)).flatten()

        return self.__get_observation()

    def render(self, mode='human', close=False):
        import matplotlib.pyplot as plt
        plt.ion()
        past_t, past_o, _ = zip(*self.__get_ohlcv_range(0, self.steps))
        curr_t, curr_o, _ = zip(*self.__get_ohlcv_range(self.steps - 1, self.steps + self.n_obs))
        futr_t, futr_o, _ = zip(*self.__get_ohlcv_range(self.steps + self.n_obs - 1, len(self.OHLCV_window['open'])))

        plt.cla()
        plt.plot(past_t, past_o, color="grey")
        plt.plot(curr_t, curr_o, color="blue")
        plt.plot(futr_t, futr_o, color="grey")
        plt.axvline(x=curr_t[0], color="grey")
        plt.axvline(x=curr_t[-1], color="grey")
        if self.buy_timestep is not None:
            plt.scatter([self.buy_timestep], [self.buy_value], color="red")
        if self.done:
            t, o, _, _, _, _ = self.__get_ohlcv_index(self.steps + self.n_obs - 1)
            plt.scatter([t], [o], color="green")
            plt.title(f'Bought at {self.buy_value}, sold at {o} for a profit of {round((o - self.buy_value) / self.buy_value, 5)}%')
            plt.draw()
            plt.pause(0.001)
            plt.waitforbuttonpress()
        else:
            plt.title(f'Bought at {self.buy_timestep} for {self.buy_value}' if self.buy_timestep is not None else 'No Trade Made')
            plt.draw()
            plt.pause(0.001)
            # time.sleep(0.1)
        print("Rendering!")

        # plt.waitforbuttonpress()

        return True

    def __get_observation(self):
        self.normalized_window = self.normalized_window[self.ohlcv_size:]
        next_timestep = self.steps + self.n_obs - 1
        ohlcv_values = self.__get_ohlcv_index(next_timestep)
        timestamp, open_p, high, low, _, volume = ohlcv_values
        self.current_price = open_p
        self.normalized_window = np.concatenate((self.normalized_window, self.OHLCV_normalizer.transform([[timestamp, open_p, volume]]).flatten()), axis=None)

        state = np.array([
            self.steps,
            1 if self.buy_timestep is not None else -1,
            self.buy_value if self.buy_value is not None else -1,
            self.current_price
        ])
        self.state_normalizer.partial_fit([state])
        normalized_state = self.state_normalizer.transform([state]).flatten()
        obs = np.concatenate((normalized_state, self.normalized_window), axis=None)

        return obs

    def __get_ohlcv_range(self, start, end):
        return np.array([[self.OHLCV_window[key][i] for key in ['timestamps', 'open', 'volume']] for i in range(start, end)])

    def __get_ohlcv_index(self, i):
        return np.array([self.OHLCV_window[key][i] for key in self.OHLCV_window.keys()])


if __name__ == "__main__":
    scaler = StandardScaler()
    print(scaler.fit_transform([[1, 2, 3, 4], [2, 2, 2, 2], [7, 9, -1, 3]]))
    print(scaler.mean_)
