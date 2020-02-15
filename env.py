# Python imports
import random
import time
from abc import abstractmethod, ABC
from collections import defaultdict
from enum import Enum

import numpy as np

from gym import Env, spaces

from market_data import HistoricalInstrumentDataset, RealtimeInstrumentData
from sklearn.preprocessing import StandardScaler, MinMaxScaler

OHLCV_KEYS = ('timestamps', 'open', 'high', 'low', 'close', 'volume')


class MarketEnv(Env, ABC):
    def __init__(self, trade_fee=0.15, ep_len=10000, min_points=100000, realtime=False, symbol=None, mode='train',
                 n_obs=100):
        self.mode = mode
        self.episode_length = ep_len
        self.trade_fee = trade_fee
        self.n_obs = n_obs
        self.steps = 0
        self.obs_dims = None
        self.episode_data_window = None
        self.episode_data_size = self.episode_length + self.n_obs
        self.realtime = realtime
        self.obs_keys = OHLCV_KEYS
        self.done = False

        import matplotlib.pyplot as plt
        self.plt = plt
        self.plt.ion()

        if self.realtime:
            assert mode != 'train', "We can't train on realtime data"
            self.data_interface = RealtimeInstrumentData(symbol=symbol, window_size=self.n_obs)
        else:
            self.data_interface = HistoricalInstrumentDataset(min_points=min_points)

        self.price_normalizer = StandardScaler()
        self.volume_normalizer = StandardScaler()
        self.timestamp_normalizer = StandardScaler()
        self.capital_normalizer = MinMaxScaler(feature_range=(0, 1000000))
        self.normalizer_map = {}

        self.normalized_obs = np.array([])

    def render(self, mode='human', close=False):
        self._render_data()
        return self._render_state()

    def _get_data_window(self, start, end, normalize=False, keys=None):
        return np.array([self._get_data_index(i, normalize=normalize, keys=keys) for i in range(start, end)])

    def _get_data_index(self, i, normalize=False, keys=None):
        keys = self.episode_data_window.keys() if keys is None else keys
        if not normalize:
            return np.array([self.episode_data_window[key][i] for key in keys])
        else:
            return np.array(
                [self.normalizer_map[key].transform([[self.episode_data_window[key][i]]]).item() for key in keys])

    def _update_window(self):
        if self.realtime:
            self.episode_data_window, new_minute_step = self.data_interface.retrieve_latest_window()
            if not new_minute_step:
                intra_minute_price_info = self._get_data_index(self.steps + self.n_obs - 1, normalize=True,
                                                               keys=self.obs_keys)
                self.normalized_obs[-1] = np.array(list(intra_minute_price_info) + self._get_normalized_state_data())
                return

        self.steps += 1
        next_timestep = self.steps + self.n_obs - 1
        self.normalized_obs = self.normalized_obs[1:]
        normalized_ohlcv = self._get_data_index(next_timestep, normalize=True, keys=self.obs_keys)

        step = np.array(list(normalized_ohlcv) + self._get_normalized_state_data())
        self.normalized_obs = np.append(self.normalized_obs, [step]).reshape((self.n_obs, self.obs_dims))

    def __normalize_new_window(self):
        norm_size = self.episode_data_size if not self.realtime else self.n_obs
        timestamps, closes, volumes = zip(
            *self._get_data_window(0, norm_size, normalize=False, keys=('timestamps', 'close', 'volume')))

        self.volume_normalizer = StandardScaler()
        fit_window = [[volume] for volume in volumes]
        self.volume_normalizer.fit(fit_window)

        self.price_normalizer = StandardScaler()
        fit_window = [[close] for close in closes]
        self.price_normalizer.fit(fit_window)

        fit_window = [[timestamp] for timestamp in timestamps]
        self.timestamp_normalizer.partial_fit(fit_window)

        self.__set_normalizer_map()

        start_window = [list(price_obs) + self._get_normalized_state_data() for price_obs in
                        self._get_data_window(0, self.n_obs, normalize=True, keys=self.obs_keys)]

        self.normalized_obs = np.array(start_window).reshape((self.n_obs, self.obs_dims))

    def _reset_data_interface(self):
        if self.realtime:
            self.episode_data_window = self.data_interface.init_window()
        else:
            self.episode_data_window = self.data_interface.get_random_window(n_points=self.episode_data_size,
                                                                             split=self.mode)
        self.__normalize_new_window()

        return self._get_observation()

    def _render_data(self):
        self.plt.cla()
        plot_keys = ('timestamps', 'close')

        if self.steps > 0:
            past_t, past_o = zip(*self._get_data_window(0, self.steps + 1, keys=plot_keys))
            self.plt.plot(past_t, past_o, color="grey", zorder=1)

        if not self.realtime:
            futr_t, futr_o = zip(
                *self._get_data_window(self.steps + self.n_obs, len(self.episode_data_window['close']),
                                       keys=plot_keys))
            self.plt.plot(futr_t, futr_o, color="grey", zorder=1)

        curr_t, curr_o = zip(*self._get_data_window(self.steps, self.steps + self.n_obs, keys=plot_keys))
        self.plt.plot(curr_t, curr_o, color="blue", zorder=1)

        self.plt.axvline(x=curr_t[0], color="black", lw=0.2)
        self.plt.axvline(x=curr_t[-1], color="black", lw=0.2)

    def _get_observation(self):
        return self.normalized_obs

    def __set_normalizer_map(self):
        self.normalizer_map = {
            'timestamps': self.timestamp_normalizer,
            'open': self.price_normalizer,
            'high': self.price_normalizer,
            'low': self.price_normalizer,
            'close': self.price_normalizer,
            'volume': self.volume_normalizer,
            'capital': self.capital_normalizer
        }

    @abstractmethod
    def _get_normalized_state_data(self):
        return []

    @abstractmethod
    def _render_state(self):
        pass


class ProfitEnv(MarketEnv):
    def __init__(self, trade_fee=0.075, n_obs=200, ep_len=10000, mode='train', min_points=100000,
                 realtime=False, symbol=None, init_capital=50000, action_granularity=10, min_trade_val=100):
        super().__init__(trade_fee=trade_fee, ep_len=ep_len, realtime=realtime, mode=mode, min_points=min_points, symbol=symbol,
                         n_obs=n_obs)

        # long_amt, short_amt, exit_short_amt, exit_long_amt
        self.action_space = spaces.MultiDiscrete([2] * 2 + [action_granularity] * 2)
        self.action_granularity = action_granularity

        self.obs_dims = 4 + len(self.obs_keys)
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.n_obs, self.obs_dims,), dtype='float32')

        self.init_capital = init_capital
        self.min_trade_val = min_trade_val
        self.curr_timestamp = 0
        self.shorted_value, self.shorted_amt, self.longed_value, self.longed_amt, self.capital = 0, 0, 0, 0, self.init_capital
        self.short_timestamps, self.short_enters, self.short_qtys = [], [], []
        self.long_timestamps, self.long_enters, self.long_qtys = [], [], []
        self.short_exit_timestamps, self.short_exits, self.short_exit_qtys = [], [], []
        self.long_exit_timestamps, self.long_exits, self.long_exit_qtys = [], [], []

        self.reset()

    def step(self, action):
        long, exit_long, long_amt, exit_long_amt = action
        self.curr_timestamp, close_p = self._get_data_index(self.steps + self.n_obs - 1, keys=('timestamps', 'close'))
        hold = sum([long, exit_long]) == 0
        long, exit_long = (x == 1 for x in (long, exit_long))
        self.longed_value = self.longed_amt * close_p

        reward = 0

        if close_p <= 0 or hold:
            pass

        if exit_long:
            exit_val = min(self.longed_value, self.init_capital * (self.action_granularity - exit_long_amt) / self.action_granularity)
            taxed_exit_val = exit_val * (1 - self.trade_fee / 100)
            exit_amt = exit_val / close_p
            if exit_val >= self.min_trade_val:
                self.long_exit_timestamps.append(self.curr_timestamp)
                self.long_exits.append(close_p)
                self.long_exit_qtys.append(exit_amt)
                self.longed_amt -= exit_amt
                self.longed_value -= exit_val
                self.capital += taxed_exit_val
                if self.longed_value == 0:
                    reward = self.capital / self.init_capital - 1
                # print(f"Exiting {round(exit_amt, 5)} at {round(exit_val, 5)}")
        elif long:
            enter_cap = min(self.capital, self.init_capital * (self.action_granularity - long_amt) / self.action_granularity)
            taxed_enter_cap = enter_cap * (1 - self.trade_fee / 100)
            n_stocks = taxed_enter_cap / close_p
            if enter_cap >= self.min_trade_val:
                self.long_timestamps.append(self.curr_timestamp)
                self.long_enters.append(close_p)
                self.long_qtys.append(enter_cap)
                self.longed_amt += n_stocks
                self.longed_value += taxed_enter_cap
                self.capital -= enter_cap
                # print(f"Longing {round(enter_cap, 5)} at {round(n_stocks, 5)}")

        # if exit_short:
        #     pass
        # if short:
        #     pass
        if self.steps == self.episode_length:
            pass

        # reward = (self.capital + self.longed_value) / self.init_capital - 1
        # liquid_pen = 0.005 * self.capital / self.init_capital  # Punish for having funds not invested
        # reward -= liquid_pen

        print(f"Capital:\t{int(self.capital)}\tStockVal:\t{int(self.longed_value)}\tAction:\t{action}\tRew:\t{reward}")

        self._update_window()

        return self._get_observation(), reward, self.steps == self.episode_length, {}

    def reset(self):
        self.steps, self.curr_timestamp, self.done = 0, 0, False
        self.shorted_value, self.shorted_amt, self.longed_value, self.longed_amt, self.capital = 0, 0, 0, 0, self.init_capital
        self.short_timestamps, self.short_enters, self.short_qtys = [], [], []
        self.long_timestamps, self.long_enters, self.long_qtys = [], [], []
        self.short_exit_timestamps, self.short_exits, self.short_exit_qtys = [], [], []
        self.long_exit_timestamps, self.long_exits, self.long_exit_qtys = [], [], []

        return self._reset_data_interface()

    def _render_state(self):
        pass

    def _get_normalized_state_data(self):
        return [val / self.init_capital for val in (self.capital, self.longed_value)] + \
                [1 if len(self.long_timestamps) > 0 and self.curr_timestamp == self.long_timestamps[-1] else -1,
                 1 if len(self.long_exit_timestamps) > 0 and self.curr_timestamp == self.long_exit_timestamps[-1] else -1]


class SingleTradeEnv(MarketEnv):
    def __init__(self, trade_fee=0.2, n_obs=100, ep_len=150, mode='train', min_points=50000,
                 realtime=False, symbol=None):
        super().__init__(trade_fee=trade_fee, ep_len=ep_len, realtime=realtime, mode=mode, min_points=min_points, symbol=symbol,
                         n_obs=n_obs)

        # long, short, hold, exit
        self.action_space = spaces.Discrete(4)

        self.obs_dims = 3 + (5 if self.obs_keys is None else len(self.obs_keys))
        self.observation_space = spaces.Box(low=-3, high=3, shape=(self.n_obs, self.obs_dims,), dtype='float32')

        self.enter_timestamp, self.enter_value, self.normalized_enter_value, self.long = None, None, None, None

        self.reset()

    # noinspection PyTypeChecker
    def step(self, action):
        long, short, exit, hold = [action == a for a in range(0, 4)]
        timestamp, close_p = self._get_data_index(self.steps + self.n_obs - 1, keys=('timestamps', 'close'))

        reward = 0
        if close_p <= 0 or hold:
            if self.enter_timestamp is not None:
                reward = -0.002
                print(f"Penalizing for hold after enter: {reward}")
            print("Holding")
        elif long or short:
            if self.enter_timestamp is None:
                print(f"Entering {'long' if long else 'short'} at {timestamp} for {close_p}")
                self.enter_timestamp, self.enter_value, self.long = timestamp, close_p, long
                self.normalized_enter_value = self.price_normalizer.transform([[self.enter_value]]).item()
            else:
                print("Trying to enter trade when already entered")
                reward = -0.2
        elif exit:
            if self.enter_timestamp is not None:
                reward = 100 * (close_p - self.enter_value) / self.enter_value
                if not self.long:
                    reward = -reward
                reward -= self.trade_fee
                print(f"Exiting after {'Long' if self.long else 'Short'}. Reward: {reward}")
                self.done = True
            else:
                print("Trying to exit when not yet entered")
                reward = -0.2
        if self.steps == self.episode_length:
            if self.enter_timestamp is not None:
                print("Ended episode with trade in progress")
            else:
                print("Ended episode without enter")

        self._update_window()

        return self._get_observation(), reward, self.steps == self.episode_length or self.done, {}

    def reset(self):
        self.done = False
        self.enter_timestamp, self.enter_value, self.normalized_enter_value = None, None, None
        self.steps = 0

        return self._reset_data_interface()

    def _render_state(self):
        plot_keys = ('timestamps', 'close')
        if self.enter_timestamp is not None:
            self.plt.scatter([self.enter_timestamp], [self.enter_value], color="red", marker="^" if self.long else "v",
                             s=50, zorder=2)
            self.plt.axhline(y=self.enter_value, color="red", lw=0.5)
        if self.done:
            timestamp, close = self._get_data_index(self.steps + self.n_obs - 1, keys=plot_keys)
            self.plt.scatter([timestamp], [close], color="green", zorder=2)
            profit = round(
                (100 if self.long else -100) * (close - self.enter_value) / self.enter_value - self.trade_fee, 4)
            self.plt.title(
                f'{"Long" if self.long else "Short"} at {self.enter_value}, exit at {close} for a profit of {profit}%')
            self.plt.draw()
            self.plt.pause(0.001)
            self.plt.waitforbuttonpress()
        else:
            self.plt.title(
                f'{"Long" if self.long else "Short"} at {self.enter_timestamp} for {self.enter_value}' if self.enter_timestamp is not None else 'No Trade Made')
            self.plt.draw()
            self.plt.pause(0.001)

        return True

    def _get_normalized_state_data(self):
        long = 1 if self.long else -1
        return [-1, -1, -1] if self.enter_value is None else [self.normalized_enter_value, long, -long]


if __name__ == "__main__":
    pass
