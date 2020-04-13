import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self):
        self._id = str(random.random()) + str(random.random())[-10:]
        self.is_terminal = False
        self.reward = 0.
        self.gradient = 0.
        self.total_q = 0.
        self.visits = 0.
        self.action = 0.


class GRABSingleAction:
    def __init__(self, env, gamma=.99, alpha=0.1, horizon=200):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.horizon = horizon

        self.bandits = [Bandit() for _ in range(horizon)]

    def do_explore_temperature(self, n):
        return np.random.random_sample() < max(.1, .999 ** n)

    def do_explore_stepwise(self, n):
        do_sample_random = np.random.random_sample() < .99
        if n > 1000:
            do_sample_random = np.random.random_sample() < .8
        if n > 2000:
            do_sample_random = np.random.random_sample() < .5
        return do_sample_random

    def calc_noise(self, visits, decay=.9995):
        return (np.random.random_sample() - .5) * (2. * decay ** visits)

    def do_exploit(self, bandit):
        return bandit.gradient

    def do_exploit_with_noise(self, bandit, noise_range=.05):
        return bandit.gradient + (np.random.random_sample() - .5) * noise_range

    def do_exploit_in_direction(self, bandit, direction_factor=.01):
        # Go into the direction of the gradient!
        # Don't just sample randomly around the current gradient value like in
        # do_exploit_with_noise.
        return bandit.gradient + np.sign(bandit.gradient) * direction_factor

    def simulate(self, total_depth, n):
        actions = []
        rewards = []
        env = copy.deepcopy(self.env)

        do_sample_random = self.do_explore_temperature(n)

        for bandit in self.bandits:
            if do_sample_random:
                action = bandit.gradient + self.calc_noise(bandit.visits)
            else:
                action = self.do_exploit(bandit)
            action = np.clip(action, -1., 1.)
            obs, reward, done, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)

            if done:
                break
        return actions, rewards

    def calc_delta_factor(self, power):
        # TODO
        # return .15 * max(.2, abs(power) / 100.)
        # return max(.1, abs(power) / 100.)
        return .5 * max(.2, abs(power) / 100. * 2.)

    def backup(self, actions, rewards, total_path_len, n):
        curr_reward = 0
        for i in range(len(rewards))[::-1]:
            curr_reward = rewards[i] + self.gamma * curr_reward

            self.bandits[i].total_q = (
                self.bandits[i].visits
                * self.bandits[i].total_q
                + curr_reward) / (self.bandits[i].visits + 1)
            self.bandits[i].visits += 1

            delta = actions[i] - self.bandits[i].gradient

            baseline = self.bandits[i].total_q
            power = curr_reward - baseline
            delta *= self.calc_delta_factor(power)
            if power == 0:
                power = (np.random.random_sample() - .5) * .001
                # power = (np.random.random_sample() - .5) * .01

            delta = np.clip(delta, -1., 1.)
            direction = np.sign(power)
            # ---- DEBUG ----
            if i == 0:
                plt.plot([b.gradient for b in self.bandits])
                plt.plot([1. for b in self.bandits])
                plt.plot([-1. for b in self.bandits])
                x = np.array(
                    [np.sign(b.gradient) * b.gradient ** 2
                        for b in self.bandits]
                )
                x = np.cumsum(x)
                x = 2. * (x - np.min(x)) / np.ptp(x) - 1
                plt.plot(x)
                plt.draw()
                plt.pause(0.01)
                plt.clf()
            # ---- DEBUG ---- E
            self.bandits[i].gradient += direction * delta
            self.bandits[i].gradient = np.clip(
                self.bandits[i].gradient, -1., 1.
            )

    def run(self, n_iter=2000, max_actions=100):
        plt.ion()
        n_iter = 1000
        self.alpha = .1
        self.baseline = 0.
        self.gamma = .999

        actions = []
        start_time = time.time()
        total_depth = 0
        for j in range(max_actions):
            for i in range(n_iter):
                action_seq, rewards = self.simulate(total_depth, i)
                self.backup(action_seq, rewards, total_depth, i)
            total_depth += 1

            actions.append(self.bandits[0].gradient)

            # ---- DEBUG ----
            x = np.array(
                [np.sign(b.gradient) * b.gradient ** 2 for b in self.bandits]
            )
            plt.plot(np.cumsum(x))
            import pdb
            pdb.set_trace()
            # ---- DEBUG ---- E

            obs, reward, done, _ = self.env.step(self.bandits[0].gradient)

            if done:
                break
        return actions, time.time() - start_time


if __name__ == "__main__":
    import gym
    import mini_continuous_env  # noqa: F401
    env = gym.make('MiniContinuousEnv-v0')
    env.reset()

    mcts_obj = GRABSingleAction(env)
    path, timing = mcts_obj.run()
    print("RESULT", path, timing)
