import matplotlib
import time

import multiprocessing
import numpy as np
from scipy.stats import binned_statistic

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from env import MarketOHLCVEnv
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


class TraderAgent:
    def __init__(self, market_env, n_env=multiprocessing.cpu_count(), mode='train'):
        self.base_env = market_env
        self.n_env = n_env
        self.env = SubprocVecEnv([lambda: market_env(mode=mode) for _ in range(n_env)])
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
        self.model = PPO2(policy, self.env, verbose=1, gamma=gamma, n_steps=int(256 / self.n_env), learning_rate=lr)

    def learn(self, timesteps, callback=None):
        if self.model is None:
            self.new_model()
        self.model.learn(total_timesteps=timesteps, callback=callback)

    def demo(self, timestep_sleep=0.2):
        env = self.base_env()
        obs = env.reset()
        while True:
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            # time.sleep(timestep_sleep)
            if done:
                print("====================")
                obs = env.reset()

    def evaluate(self, n_episodes=100):
        assert self.mode == 'test', "Must be in test mode for evaluation"
        episode_rewards = []
        env = self.base_env()
        obs = env.reset()
        for i in range(n_episodes):
            action, _states = self.model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                episode_rewards.append(reward)
                obs = env.reset()


#
# lp = LossPlotter()
# def learn_callback():
#     pass


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


def cnn_extractor(scaled_images, channels=1, w=6, h=50):
    print(f"========= REAL SHAPE: {scaled_images.shape} ===========")
    original_shape = scaled_images.shape[1]
    print(f"========= SHAPE: {original_shape} ===========")
    scaled_images = tf.reshape(scaled_images, (-1, h, w,  channels))
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=w, stride=1, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=1, stride=1, init_scale=np.sqrt(2)))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=128, filter_size=1, stride=1, init_scale=np.sqrt(2)))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))


def run_train(name, load=False):
    ppo_agent = TraderAgent(MarketOHLCVEnv, n_env=8, mode='train')
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


def run_demo(name):
    ppo_agent = TraderAgent(MarketOHLCVEnv, n_env=1, mode='test')
    ppo_agent.load_model("models/" + name)
    ppo_agent.demo(timestep_sleep=0.)


if __name__ == "__main__":
    name = "trader_conv_50-pen_vol"

    # run_train(name, load=True)
    run_demo(name)
