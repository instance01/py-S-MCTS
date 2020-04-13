import time
import copy
import numpy as np
from node import Node


class MCTS:
    def __init__(self, env, gamma=.99, c=.2):
        self.Q = {}
        self.visits = {}

        env.reset()
        self.env = copy.copy(env)

        self.root_node = Node()
        self.root_node.env = copy.deepcopy(env)
        self.root_node.calc_hash()
        self.Q[self.root_node] = 0
        self.visits[self.root_node] = 0

        self.gamma = gamma
        self.c = c

    def _ucb(self, parent_node, child_node):
        mean_q = self.Q[child_node] / self.visits[child_node]
        expl = 2 * np.log(self.visits[parent_node]) / self.visits[child_node]
        return mean_q + self.c * expl ** .5

    def _gen_children_nodes(self, parent_node):
        for action in parent_node.env.actions:
            env = copy.copy(parent_node.env)
            obs, reward, done, _ = env.step(action)
            node = Node()
            node.env = env
            node.action = action
            node.reward = reward
            node.is_terminal = done
            node.parent = parent_node
            node.calc_hash()
            parent_node.children.append(node)
            self.Q[node] = 0.
            self.visits[node] = 0

    def _expand(self, parent_node):
        """Pick the first action that was not visited yet.
        """
        if len(parent_node) == 0:
            self._gen_children_nodes(parent_node)

        for i, child_node in enumerate(parent_node):
            if self.visits[child_node] == 0:
                return i, child_node

        parent_node.is_fully_expanded = True

        return 0, None

    def _get_best_node(self, parent_node):
        # TODO Support not only UCB, but also eps greedy and Boltzmann.
        children = []
        max_ucb = max(
            self._ucb(parent_node, child_node) for child_node in parent_node
        )
        for i, child_node in enumerate(parent_node):
            ucb_child = self._ucb(parent_node, child_node)
            if ucb_child >= max_ucb:
                children.append((i, child_node))
        return children[np.random.choice(len(children))]

    def select_expand(self):
        """Select best node until finding a node that is not fully expanded.
        Expand it and return the expanded node (together with length of path
        for gamma).
        """
        path_len = 0

        curr_node = self.root_node
        while True:
            if curr_node.is_terminal:
                break
            if curr_node.is_fully_expanded:
                _, curr_node = self._get_best_node(curr_node)
                path_len += 1
            else:
                _, node = self._expand(curr_node)
                if node is not None:
                    path_len += 1
                    return node, path_len
        return curr_node, path_len

    def simulate(self, curr_node, depth=1):
        if curr_node.is_terminal:
            return curr_node.reward
        env = copy.copy(curr_node.env)
        q_val = 0.
        i = depth
        while True:
            action = np.random.choice(env.actions)
            obs, reward, done, _ = env.step(action)

            q_val += self.gamma**i * reward
            i += 1
            if done:
                break
        return q_val

    def backup(self, curr_node, q_val, total_path_len):
        while curr_node is not None:
            total_path_len -= 1
            discount = self.gamma ** total_path_len
            q_val += curr_node.reward * discount
            self.Q[curr_node] += q_val
            self.visits[curr_node] += 1
            curr_node = curr_node.parent

    def run(self, n_iter=2000, max_actions=100):
        actions = []
        start_time = time.time()
        total_depth = 0
        for j in range(max_actions):
            for i in range(n_iter):
                node, path_len = self.select_expand()
                q_val = self.simulate(node, path_len + total_depth)
                self.backup(node, q_val, path_len + total_depth)
            total_depth += 1

            curr_node = self.root_node
            curr_node = self._get_best_node(curr_node)[1]
            self.env.step(curr_node.action)
            self.root_node = curr_node
            curr_node.parent = None

            actions.append(curr_node.action)

            if curr_node.is_terminal:
                break
        return actions, time.time() - start_time
