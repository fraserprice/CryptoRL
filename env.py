# Python imports
import random
import time
from collections import defaultdict
from enum import Enum

import numpy as np

from gym import Env, spaces

from market_data import InstrumentDataset
from sklearn.preprocessing import StandardScaler


class MarketOHLCVEnv(Env):
    def __init__(self, trade_fee=0.002, n_obs=50, ep_len=150, mode='train', min_points=50000, obs_keys=['timestamps', 'open', 'volume']):
        import matplotlib.pyplot as plt
        self.mode = mode
        self.instrument_data = InstrumentDataset(min_points=min_points)
        self.episode_length = ep_len
        self.trade_fee = trade_fee

        self.n_obs = n_obs
        self.obs_keys = obs_keys
        self.obs_dims = 3 + (5 if obs_keys is None else len(obs_keys))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.n_obs, self.obs_dims,), dtype='float32')

        self.price_normalizer = StandardScaler()
        self.volume_normalizer = StandardScaler()
        self.timestamp_normalizer = StandardScaler()
        self.normalizer_map = {}

        self.steps = 0
        self.done = False
        self.enter_timestamp, self.enter_value, self.normalized_enter_value, self.long = None, None, None, None
        self.instrument = None
        self.episode_data_window = None
        self.episode_data_size = self.episode_length + self.n_obs
        self.normalized_obs = np.array([])

        self.plt = plt
        self.plt.ion()

        self.reset()

    # noinspection PyTypeChecker
    def step(self, action):
        long, short, exit, hold = [action == a for a in range(0, 4)]
        timestamp, open_p = self.__get_ohlcv_index(self.steps + self.n_obs - 1, keys=['timestamps', 'open'])

        reward = 0
        if open_p <= 0 or hold:
            if self.enter_timestamp is not None:
                reward = -0.001
                print(f"Penalizing for hold after enter: {reward}")
            print("Holding")
        elif long or short:
            if self.enter_timestamp is None:
                print(f"Entering {'long' if long else 'short'} at {timestamp} for {open_p}")
                self.enter_timestamp, self.enter_value, self.long = timestamp, open_p, long
                self.normalized_enter_value = self.price_normalizer.transform([[self.enter_value]]).item()
            else:
                print("Trying to enter trade when already entered")
                reward = -0.1
        elif exit:
            if self.enter_timestamp is not None:
                reward = 100 * (open_p - self.enter_value) / self.enter_value
                if not long:
                    reward = -reward
                print(f"Exiting after {'Long' if self.long else 'Short'}. Reward: {reward}")
                self.done = True
            else:
                print("Trying to exit when not yet entered")
                reward = -0.1
        if self.steps == self.episode_length:
            if self.enter_timestamp is not None:
                print("Ended episode with trade in progress")
            else:
                print("Ended episode without enter")

        self.steps += 1
        next_timestep = self.steps + self.n_obs - 1

        self.normalized_obs = self.normalized_obs[1:]
        normalized_ohlcv = self.__get_ohlcv_index(next_timestep, normalize=True, keys=self.obs_keys)

        if self.enter_value is None:
            step = np.array(list(normalized_ohlcv) + [-1, -1, -1])
        else:
            long = 1 if self.long else -1
            step = np.array(list(normalized_ohlcv) + [self.normalized_enter_value, long, -long])
        self.normalized_obs = np.append(self.normalized_obs, [step]).reshape((self.n_obs, self.obs_dims))

        return self.__get_observation(), reward, self.steps == self.episode_length or self.done, {}

    def reset(self):
        self.done = False
        self.enter_timestamp, self.enter_value, self.normalized_enter_value = None, None, None
        self.episode_data_window = self.instrument_data.get_random_window(n_points=self.episode_data_size,
                                                                          split=self.mode)
        self.steps = 0
        self.__get_new_random_window()

        return self.__get_observation()

    def render(self, mode='human', close=False):
        self.plt.cla()
        past_t, past_o = zip(*self.__get_ohlcv_window(0, self.steps, keys=['timestamps', 'open']))
        self.plt.plot(past_t, past_o, color="grey")
        curr_t, curr_o = zip(*self.__get_ohlcv_window(self.steps - 1, self.steps + self.n_obs, keys=['timestamps', 'open']))
        futr_t, futr_o = zip(*self.__get_ohlcv_window(self.steps + self.n_obs - 1, len(self.episode_data_window['open']), keys=['timestamps', 'open']))

        self.plt.plot(curr_t, curr_o, color="blue")
        self.plt.plot(futr_t, futr_o, color="grey")

        self.plt.axvline(x=curr_t[0], color="grey" if not self.done else "green")
        self.plt.axvline(x=curr_t[-1], color="grey" if not self.done else "green")
        if self.enter_timestamp is not None:
            self.plt.scatter([self.enter_timestamp], [self.enter_value], color="red", marker="^" if self.long else "v")
            self.plt.axhline(y=self.enter_value, color="red", lw=0.5)
        if self.done:
            timestamp, open = self.__get_ohlcv_index(self.steps + self.n_obs - 1, keys=['timestamps', 'open'])
            self.plt.scatter([timestamp], [open], color="green")
            profit = round((100 if self.long else -100) * (open - self.enter_value) / self.enter_value, 5)
            self.plt.title(f'{"Long" if self.long else "Short"} at {self.enter_value}, exit at {open} for a profit of {profit}%')
            self.plt.draw()
            self.plt.pause(0.001)
            self.plt.waitforbuttonpress()
        else:
            self.plt.title(
                f'{"Long" if self.long else "Short"} at {self.enter_timestamp} for {self.enter_value}' if self.enter_timestamp is not None else 'No Trade Made')
            self.plt.draw()
            self.plt.pause(0.001)

        return True

    def __get_observation(self):
        obs = self.normalized_obs

        return obs

    def __get_ohlcv_window(self, start, end, normalize=False, keys=None):
        return np.array([self.__get_ohlcv_index(i, normalize=normalize, keys=keys) for i in range(start, end)])

    def __get_ohlcv_index(self, i, normalize=False, keys=None):
        keys = self.episode_data_window.keys() if keys is None else keys
        if not normalize:
            return np.array([self.episode_data_window[key][i] for key in keys])
        else:
            return np.array([self.normalizer_map[key].transform([[self.episode_data_window[key][i]]]).item() for key in keys])

    def __get_new_random_window(self):
        timestamps, opens, volumes = zip(*self.__get_ohlcv_window(0, self.episode_data_size, normalize=False, keys=['timestamps', 'open', 'volume']))

        self.volume_normalizer = StandardScaler()
        fit_window = [[volume] for volume in volumes]
        self.volume_normalizer.fit(fit_window)

        self.price_normalizer = StandardScaler()
        fit_window = [[open] for open in opens]
        self.price_normalizer.fit(fit_window)

        fit_window = [[timestamp] for timestamp in timestamps]
        self.timestamp_normalizer.partial_fit(fit_window)

        self.__set_normalizer_map()

        start_window = [list(price_obs) + [-1, -1, -1] for price_obs in self.__get_ohlcv_window(0, self.n_obs, normalize=True, keys=self.obs_keys)]

        self.normalized_obs = np.array(start_window).reshape((self.n_obs, self.obs_dims))

    def __set_normalizer_map(self):
        self.normalizer_map = {
            'timestamps': self.timestamp_normalizer,
            'open': self.price_normalizer,
            'high': self.price_normalizer,
            'low': self.price_normalizer,
            'close': self.price_normalizer,
            'volume': self.volume_normalizer
        }


if __name__ == "__main__":
    scaler = StandardScaler()
    print(scaler.fit_transform([[1, 2, 3, 4], [2, 2, 2, 2], [7, 9, -1, 3]]))
    print(scaler.mean_)
