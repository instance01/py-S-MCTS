from enum import Enum
import pickle
import time

import gym
import gym_minigrid  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import mcts
import smcts


class StatePenalty(gym.core.Wrapper):
    """Adds a penalty for each state visited, effectively punishing long paths.
    """
    def __init__(self, env):
        super(StatePenalty, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if reward == 0:
            reward = -.01

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def patch_hash_func_for_grid_world():
    # Monkey-patch Node __hash__ based on environment.
    def calc_hash(cls):
        # Calculate hash using Cantor pairing and cache it.
        p = cls.env.agent_pos
        d = cls.env.agent_dir
        p = p[0] * 10 + p[1]
        cls._hash = int(.5 * (p + d) * (p + d + 1) + d)
    mcts.Node.calc_hash = calc_hash


class Experiment1:
    """Run MCTS on empty grid world of size 5x5 (effectively 3x3 due to walls).
    """
    def __init__(self):
        self.n_iter = 1000
        self.runs = 1
        self.env = gym.make('MiniGrid-Empty-5x5-v0')
        self.env.actions = self.Action

        patch_hash_func_for_grid_world()

        self.report = []

    class Action(Enum):
        left = 0
        right = 1
        forward = 2

    def test1(self):
        """ TODO Comment
        """
        params = [
            (.95, .4), (.95, .3), (.95, .2), (.97, .4), (.97, .3),
            (.8, .5), (.8, .4), (.8, .3), (.83, .5), (.83, .4), (.83, .3)
        ]
        for param in params:
            path_lengths = []
            timings = []
            for _ in range(self.runs):
                mcts_obj = mcts.MCTS(self.env, *param)
                path, timing = mcts_obj.run(self.n_iter)
                path_lengths.append(len(path))
                timings.append(timing)
            self.report.append((path_lengths, timings, param))

    def run(self):
        self.test1()
        return self.report


class Experiment2:
    """Run MCTS on empty grid world of size 8x8.
    """
    def __init__(self):
        self.n_iter = 2000
        self.runs = 10
        self.env = gym.make('MiniGrid-Empty-8x8-v0')
        self.env.actions = self.Action

        patch_hash_func_for_grid_world()

        self.report = []

    class Action(Enum):
        left = 0
        right = 1
        forward = 2

    def test1(self):
        """ TODO Comment
        """
        params = [(.98, .2), (.98, .1), (.99, .2), (.99, .1)]
        for param in params:
            path_lengths = []
            timings = []
            for _ in range(self.runs):
                mcts_obj = mcts.MCTS(self.env, *param)
                path, timing = mcts_obj.run(self.n_iter)
                path_lengths.append(len(path))
                timings.append(timing)
            self.report.append((path_lengths, timings, param))

    def run(self):
        self.test1()
        return self.report


class Experiment3:
    """Run SMCTS on empty grid world of size 5x5.
    """
    def __init__(self):
        self.n_iter = 100
        self.runs = 50
        self.env = gym.make('MiniGrid-Empty-5x5-v0')
        self.env.actions = self.Action

        # TODO DOES NOT WORK YET.
        # self.env = StatePenalty(self.env)

        patch_hash_func_for_grid_world()

        self.report = []

    class Action(Enum):
        left = 0
        right = 1
        forward = 2

    def test1(self):
        """ TODO Comment
        """
        # Parameters: gamma, c, action coverage, err tolerance, horizon
        params = [
            # (.8, .5, .9, .1, 4),
            # (.8, .5, .8, .1, 4),
            # (.8, .5, .7, .1, 4),
            # (.8, .5, .9, .2, 4)
            (.8, .5, .99, .02, 4),
            (.8, .5, .985, .025, 4),
            (.8, .5, .98, .03, 4)
        ]
        for param in params:
            path_lengths = []
            timings = []
            for _ in range(self.runs):
                smcts_obj = smcts.SMCTS(self.env, *param)
                path, timing = smcts_obj.run(self.n_iter)
                path_lengths.append(len(path))
                timings.append(timing)
            self.report.append((path_lengths, timings, param))

    def run(self):
        self.test1()
        return self.report


# TODO More experiments here..


def run_experiment(experiment_name):
    experiment = globals()[experiment_name]()
    report = experiment.run()
    fname = 'report%s_%d.pickle' % (experiment_name, int(time.time()))
    with open(fname, 'wb+') as f:
        pickle.dump(report, f)


def load_and_plot(fname):
    with open(fname, 'rb') as f:
        report = pickle.load(f)
    plot_hist(report)


def plot_hist(report):
    # with plt.style.context('seaborn-pastel'):
    with plt.style.context('Solarize_Light2'):
        for lengths, timings, comment in report:
            # Path length histogram
            values, counts = np.unique(lengths, return_counts=True)
            plt.vlines(values, 0, counts, color='C1', lw=5)
            plt.ylim(0, max(counts) * 1.05)
            plt.xlabel('Path length')
            plt.xticks(values)
            plt.title(
                'Distribution of path lengths\n'
                '(params: %s)'
                % str(comment)
            )
            plt.show()

            # Calculation time histogram
            plt.hist(timings, bins=50)
            plt.xlabel('Seconds')
            plt.title(
                'Distribution of calculation times\n(params: %s)'
                % str(comment)
            )
            plt.show()
