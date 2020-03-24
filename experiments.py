from enum import Enum
import matplotlib.pyplot as plt
import gym_minigrid  # noqa: F401
import gym
import pickle
import mcts


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
        params = [(.95, .4), (.95, .3), (.95, .2), (.97, .4), (.97, .3)]
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


# TODO More experiments here..


def run_experiment(experiment):
    experiment = globals()[experiment]()
    report = experiment.run()
    fname = 'report' + experiment.__class__.__name__ + '.pickle'
    with open(fname, 'wb+') as f:
        pickle.dump(report, f)


def load_and_plot(fname):
    with open(fname, 'rb') as f:
        report = pickle.load(f)
    plot_hist(report)


def plot_hist(report):
    for lengths, timings, comment in report:
        plt.hist(lengths, bins=50)
        plt.title(
            'Distribution of path lengths (number of actions until goal) '
            '(params: %s)'
            % str(comment)
        )
        plt.show()
        plt.hist(timings, bins=50)
        plt.title(
            'Distribution of calculation times (params: %s)'
            % str(comment)
        )
        plt.show()
