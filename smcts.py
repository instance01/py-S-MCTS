import math
import time
import copy
import numpy as np
from mcts import MCTS
from node import Node
from collections import OrderedDict


# TODO Implementation is very similar to mcts. Notable exception is children,
# which is now a dict and subgoal related things. Still, needs generalization.


class SMCTS(MCTS):
    def __init__(
            self,
            env,
            gamma=.9,
            c=.4,
            action_coverage=.9,
            err_tolerance=.1,
            horizon=4):
        super(SMCTS, self).__init__(env)
        self.horizon = horizon
        self.threshold = math.ceil(
            math.log(err_tolerance) / math.log(action_coverage)
        )
        self.root_node.children = OrderedDict()
        self.root_node.actions = []
        self.root_node.action = None

    def _is_subgoal(self, obs):
        """TODO: This is a default _is_subgoal implementation (in this case for
        grid world).
        """
        # An observation of gym minigrid has a 7x7 viewport by default. It is
        # also in reverse, i.e. to get the grid in front of the agent we need
        # to use index -1.
        # We define the corners of the grid as subgoals. Additionally, we make
        # sure that the same corner with multiple directions is not a subgoal
        # by simply ignoring half of the possible directions.
        direction = obs['direction']
        if direction == 3 or direction == 2:
            return False
        img = obs['image']
        forward_wall = img[3][-1][0] == 1 and img[3][-2][0] == 2
        left_wall = img[2][-1][0] == 2
        right_wall = img[4][-1][0] == 2
        return forward_wall and (left_wall or right_wall)

    def _gen_node(self, parent_node, env, action, reward, done):
        node = Node()
        node.env = env
        node.action = action
        node.reward = reward
        node.is_terminal = done
        node.parent = parent_node
        node.calc_hash()
        node.children = OrderedDict()
        return node

    def _expand(self, parent_node, curr_depth, horizon=4):
        """Refer to Algorithm 1 of the SMCTS paper.
        Discover new or known macro actions until confidence is reached (the
        threshold).
        Horizon makes sure that we do not visit the goal instantly on empty
        grid world, which results in long action sets.
        """
        n = 0
        while n < self.threshold:
            actions = []
            curr_reward = 0
            env = copy.copy(parent_node.env)
            i = 0
            while True:
                i += 1
                if i > self.horizon:
                    break

                action = np.random.choice(env.actions)
                obs, reward, done, _ = env.step(action)
                # TODO Make this a wrapper. (see StatePenalty)
                # if reward == 0:
                #     reward = -0.01

                actions.append(action)
                curr_reward += reward * self.gamma ** curr_depth
                curr_depth += 1

                if self._is_subgoal(obs) or done:
                    new_node = self._gen_node(
                        parent_node,
                        copy.copy(env),
                        action,
                        curr_reward,
                        done
                    )
                    new_node.actions = actions
                    node_ = parent_node.children.get(new_node._hash)
                    if node_ is not None:
                        n += 1
                        if curr_reward > node_.reward:
                            parent_node.children[new_node._hash].actions = actions  # noqa: E501
                            parent_node.children[new_node._hash].reward = curr_reward  # noqa: E501
                        break
                    else:
                        parent_node.children[new_node._hash] = new_node
                        self.Q[new_node] = 0
                        self.visits[new_node] = 0
                        return new_node._hash

        parent_node.is_fully_expanded = True

        return None

    def _get_best_node(self, parent_node):
        # TODO Support not only UCB, but also eps greedy and Boltzmann.
        children = []
        max_ucb = max(
            self._ucb(parent_node, child_node)
            for child_node in parent_node.children.values()
        )
        for child_node in parent_node.children.values():
            ucb_child = self._ucb(parent_node, child_node)
            if ucb_child >= max_ucb:
                children.append(child_node)
        return children[np.random.choice(len(children))]

    def select_expand(self, curr_depth):
        """Select best node until finding a node that is not fully expanded.
        Expand it and return the expanded node (together with length of path
        for gamma).
        """
        path = []
        curr_node = self.root_node
        while True:
            if curr_node.is_terminal:
                break
            if curr_node.is_fully_expanded:
                curr_node = self._get_best_node(curr_node)
                path.extend(curr_node.actions)
            else:
                node_hash = self._expand(curr_node, curr_depth)
                if node_hash is not None:
                    child_node = curr_node.children[node_hash]
                    path.extend(child_node.actions)
                    return child_node, len(path)
        return curr_node, len(path)

    def backup(self, curr_node, q_val):
        curr_node_temp = copy.copy(curr_node)

        total_path_len = 0
        while curr_node is not None:
            total_path_len += len(curr_node.actions)
            curr_node = curr_node.parent

        curr_node = curr_node_temp
        while curr_node is not None:
            total_path_len -= len(curr_node.actions)
            discount = self.gamma ** total_path_len
            self.Q[curr_node] += curr_node.reward * discount
            self.visits[curr_node] += 1
            curr_node = curr_node.parent

    def run(self, n_iter=100, max_actions=100):
        actions = []
        start_time = time.time()
        total_depth = 0
        for j in range(max_actions):
            for _ in range(n_iter):
                node, path_len = self.select_expand(total_depth)
                q_val = self.simulate(node, path_len + total_depth)
                self.backup(node, q_val)

            curr_node = self.root_node
            curr_node = self._get_best_node(curr_node)
            for action in curr_node.actions:
                self.env.step(action)
            self.root_node = curr_node
            curr_node.parent = None

            total_depth += len(curr_node.actions)

            actions.extend(curr_node.actions)

            if curr_node.is_terminal:
                break
        return actions, time.time() - start_time
