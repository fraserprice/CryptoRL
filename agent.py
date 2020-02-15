import statistics

import matplotlib
from matplotlib import pyplot as plt
import time

import multiprocessing
import numpy as np
import scipy.stats as stats
import math
from scipy.stats import binned_statistic

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from env import SingleTradeEnv, ProfitEnv
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


class TraderAgent:
    def __init__(self, env_generator, n_env=multiprocessing.cpu_count(), mode='train'):
        self.base_env_generator = env_generator
        self.n_env = n_env
        self.env = SubprocVecEnv([env_generator for _ in range(n_env)])
        self.model = None
        self.mode = mode

    def load_model(self, path):
        print("Loading model")
        self.model = PPO2.load(path, self.env)

    def save_model(self, path="ppo2_simple_robot"):
        if self.model is None:
            raise AssertionError("Model does not exist- cannot be saved.")
        self.model.save(path)

    def new_model(self, policy=MlpPolicy, gamma=0.99, lr=0.00025):
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma, n_steps=int(512 / self.n_env), learning_rate=lr)

    def learn(self, timesteps, callback=None):
        if self.model is None:
            self.new_model()
        self.model.learn(total_timesteps=timesteps, callback=callback)

    def demo(self):
        env = self.base_env_generator()
        obs = env.reset()
        env.render()
        while True:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                print("====================")
                obs = env.reset()

    def evaluate(self, n_episodes=100):
        assert self.mode == 'test', "Must be in test mode for evaluation"
        episode_rewards = []
        env = self.base_env_generator()
        obs = env.reset()
        i = 0
        while True:
            print(f"Episode {i}")
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                episode_rewards.append(reward)
                obs = env.reset()
                i += 1
                if i == n_episodes:
                    break

        mu = statistics.mean(episode_rewards)
        variance = statistics.variance(episode_rewards)
        sigma = math.sqrt(variance)
        # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        # plt.plot(x, stats.norm.pdf(x, mu, sigma))
        # plt.show()
        return mu, variance


class LossPlotter:
    def __init__(self, max_points=100000000):
        from matplotlib import pyplot as plt
        self.plt = plt
        self.plt.ion()
        self.plt.show()
        self.timesteps = []
        self.rewards = []
        self.aggregated_timesteps = []
        self.aggregated_rewards = []
        fig = plt.figure()
        self.ax = fig.add_subplot()
        self.max_points = max_points
        self.bin_size = 1
        self.t_start = None
        self.last_checkpoint = None

    def add_point(self, distance):
        t = time.time()
        if self.t_start is None:
            self.t_start = t
        self.timesteps.append(t - self.t_start)
        self.rewards.append(distance)
        return t

    def plot(self):
        self.plt.xlabel("Time/s")
        self.plt.ylabel("Average reward")
        self.plt.cla()
        if len(self.timesteps) <= self.max_points:
            self.plt.plot(self.timesteps, self.rewards)
        else:
            self.aggregated_rewards, self.aggregated_timesteps, _ = binned_statistic(self.timesteps, self.rewards,
                                                                                     bins=self.max_points)
            self.plt.plot(self.aggregated_timesteps[1:], self.aggregated_rewards)
        self.plt.draw()
        self.plt.pause(0.001)

    def get_plot_callback(self, checkpoint_interval=1000, filename=None, verbose=False):
        def f(inp1, _):

            mean_reward = None
            if 'true_reward' in inp1:
                reward = np.array(inp1['true_reward'])
                zero_removed_reward = reward[reward != 0]
                mean_reward = zero_removed_reward.mean()
            if 'info' in inp1:
                print(inp1['info'])
            if mean_reward is not None:
                t = self.add_point(mean_reward)
                elapsed = 0 if self.t_start is None else time.time() - self.t_start
                if 1 <= checkpoint_interval <= elapsed - (0 if self.last_checkpoint is None else self.last_checkpoint):
                    self.last_checkpoint = elapsed
                    if not verbose:
                        matplotlib.use('Agg')
                    self.save(filename)
                if verbose:
                    self.plot()
                    self.plt.pause(0.000001)
                return t
            return None

        return f

    def save(self, filename):
        self.plot()
        self.plt.savefig(filename)


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         layers=[128, 100, 64, 32, 16], **_kwargs)


class CustomCnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         layers=[64, 32], **_kwargs, cnn_extractor=cnn_extractor)


class CustomMlpLstmPolicy(MlpLstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, layers=[48, 36, 24, 12, 6],
                         **_kwargs)


def run_demo(env_gen, name):
    ppo_agent = TraderAgent(env_gen, n_env=1)
    ppo_agent.load_model("models/" + name)
    ppo_agent.demo()


def run_eval(env_gen, name, n_episodes=10000):
    ppo_agent = TraderAgent(env_gen, n_env=1)
    ppo_agent.load_model("models/" + name)
    return ppo_agent.evaluate(n_episodes=n_episodes)


def run_train(env_gen, name, n_env=16, load=False):
    ppo_agent = TraderAgent(env_gen, n_env=n_env)
    if load:
        ppo_agent.load_model("models/" + name)
    else:
        ppo_agent.new_model(policy=CustomCnnPolicy, gamma=0.995)
    loss_plotter = LossPlotter(max_points=10000000)
    save_interval = 50000
    for i in range(0, 50000000, save_interval):
        ppo_agent.learn(save_interval, callback=loss_plotter.get_plot_callback(verbose=True, filename="figures/" + name,
                                                                               checkpoint_interval=60))
        print("Saving...")
        loss_plotter.save("figures/" + name)
        ppo_agent.save_model("models/" + name)


def run_single_trade_train(name, load=False, min_points=100000, trade_fee=0.15, n_env=16):
    env_gen = lambda: SingleTradeEnv(mode='rain',
                                     trade_fee=trade_fee,
                                     min_points=min_points)
    run_train(env_gen, name, n_env=n_env, load=load)


def run_single_trade_demo(name, min_points=50000, realtime=False, symbol=None, trade_fee=0.15):
    env_gen = lambda: SingleTradeEnv(mode='test',
                                     trade_fee=trade_fee,
                                     min_points=min_points,
                                     realtime=realtime,
                                     symbol=symbol)
    run_demo(env_gen, name)


def run_single_trade_eval(name, n_episodes=100000, trade_fee=0.15, min_points=50000):
    env_gen = lambda: SingleTradeEnv(mode='test',
                                     trade_fee=trade_fee,
                                     min_points=min_points)
    return run_eval(env_gen, name, n_episodes=n_episodes)


def run_profit_train(name, load=False, min_points=100000, trade_fee=0.15, n_env=16, init_capital=50000,
                     action_granularity=10):
    env_gen = lambda: ProfitEnv(mode='train',
                                trade_fee=trade_fee,
                                min_points=min_points,
                                init_capital=init_capital,
                                action_granularity=action_granularity)
    run_train(env_gen, name, n_env=n_env, load=load)


def run_profit_demo(name, min_points=50000, realtime=False, symbol=None, trade_fee=0.15, action_granularity=10, init_capital=50000):
    env_gen = lambda: ProfitEnv(mode='test',
                                trade_fee=trade_fee,
                                min_points=min_points,
                                realtime=realtime,
                                symbol=symbol,
                                action_granularity=action_granularity,
                                init_capital=init_capital)
    run_demo(env_gen, name)


def run_profit_eval(name, n_episodes=100000, trade_fee=0.15, min_points=50000, action_granularity=10, init_capital=50000):
    env_gen = lambda: ProfitEnv(mode='test',
                                trade_fee=trade_fee,
                                min_points=min_points,
                                action_granularity=action_granularity,
                                init_capital=init_capital)
    return run_eval(env_gen, name, n_episodes=n_episodes)


def cnn_extractor(scaled_images, channels=1, w=8, h=100):
    print(f"========= REAL SHAPE: {scaled_images.shape} ===========")
    original_shape = scaled_images.shape[1]
    print(f"========= SHAPE: {original_shape} ===========")
    scaled_images = tf.reshape(scaled_images, (-1, h, w, channels))
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=w, stride=1, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2)))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=128, filter_size=1, stride=1, init_scale=np.sqrt(2)))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))


if __name__ == "__main__":
    inp_name = "profit_tohlcv_100obs_100k-min_075fee_10k-ep_0-02-cap-pen"

    run_profit_train(inp_name, load=False, init_capital=50000, action_granularity=10, trade_fee=0.075)

    # run_single_trade_train(inp_name, load=False, min_points=100000, n_env=32)
    # run_single_trade_demo(name, realtime=False, symbol='ETHBTC')
