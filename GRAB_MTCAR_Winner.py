import random
import time
import copy
import numpy as np


# TODO Code is full of magic numbers.
# TODO Needs heavy cleanup. Priority was getting something working.


class Bandit:
    def __init__(self):
        self._id = str(random.random()) + str(random.random())[-10:]
        self.is_terminal = False
        self.reward = 0.
        self.gradient = 0.
        self.total_q = 0.
        self.visits = 0.
        self.action = 0.
        self.last_power = 0.


class GRABSingleAction:
    def __init__(self, env, gamma=.99, alpha=0.1, horizon=300):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.horizon = horizon

        self.bandits = [Bandit() for _ in range(horizon)]

    def do_explore_temperature(self, n, decay=.999):
        return np.random.random_sample() < max(.1, decay ** n)

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

        do_sample_random = self.do_explore_temperature(n, decay=.9999)

        # Go even further than the horizon (300).
        for i in range(1000):
            if i >= self.horizon:
                action = self.calc_noise(0)
                obs, reward, done, _ = env.step([action])
                actions.append(action)
                rewards.append(reward)
                if done:
                    break
                continue

            bandit = self.bandits[i]
            if do_sample_random:
                if bandit.last_power > 2.:
                    action = bandit.gradient + self.calc_noise(0, decay=.05)
                else:
                    noise = self.calc_noise(bandit.visits, decay=.9999)
                    action = bandit.gradient + noise
            else:
                action = self.do_exploit_with_noise(bandit, noise_range=.01)
            action = np.clip(action, -1., 1.)
            obs, reward, done, _ = env.step([action])

            actions.append(action)
            rewards.append(reward)

            if done:
                break
        return actions, rewards, not do_sample_random

    def calc_delta_factor(self, power):
        # 100 is magic: This is the maximum reward achievable.
        return 0.3 * max(.2, abs(power ** 1.0) / 100. ** 1.0)

    def backup(self, actions, rewards, total_path_len, n, exploited):
        curr_reward = 0
        for i in range(len(rewards))[::-1]:
            curr_reward = rewards[i] + self.gamma * curr_reward

            # Since we now go over the horizon when simulating, the rewards
            # list is also bigger. But there are no bandits to update after
            # the horizon is reached, so simply ignore and add up the rewards.
            # Keep in mind, we are going backwards here!
            # Thus we add up the rewards.
            if i >= self.horizon:
                continue

            self.bandits[i].total_q = (
                self.bandits[i].visits
                * self.bandits[i].total_q
                + curr_reward) / (self.bandits[i].visits + 1)
            self.bandits[i].visits += 1

            delta = actions[i] - self.bandits[i].gradient

            baseline = self.bandits[i].total_q
            power = curr_reward - baseline
            self.bandits[i].last_power = power
            delta *= self.calc_delta_factor(power)

            if exploited:
                delta *= 1.1

            delta = np.clip(delta, -1., 1.)
            direction = np.sign(power)
            self.bandits[i].gradient += direction * delta
            self.bandits[i].gradient = np.clip(
                self.bandits[i].gradient, -1., 1.
            )

    def run(self, n_iter=2000, max_actions=1000):
        # TODO These settings don't belong here!
        n_iter = 1100
        self.alpha = .1
        self.gamma = 1.0

        total_reward = 0.

        actions = []
        start_time = time.time()
        total_depth = 0
        for j in range(max_actions):
            for i in range(n_iter):
                action_seq, rewards, exploited = self.simulate(total_depth, i)
                self.backup(action_seq, rewards, total_depth, i, exploited)
            total_depth += 1

            actions.append(self.bandits[0].gradient)

            obs, reward, done, _ = self.env.step([self.bandits[0].gradient])

            print("TAKEN ACTION", self.bandits[0].gradient, obs, reward, done)
            total_reward += reward

            # Bandits are actually reset here!!
            # So we don't end up in a local minimum.
            # Needs more testing though.. Maybe reusing is better.
            self.bandits = [Bandit() for __ in range(self.horizon)]
            n_iter = 700

            if done:
                print("TOTAL REWARD", total_reward)
                break
        return actions, time.time() - start_time


if __name__ == "__main__":
    import gym
    env = gym.make('MountainCarContinuous-v0')
    env.reset()
    mcts_obj = GRABSingleAction(env)
    path, timing = mcts_obj.run()
    print("RESULT", path, timing)
