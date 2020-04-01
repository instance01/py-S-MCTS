import math
import time
import copy
import numpy as np
from node import Node
from mcts import MCTS


class GradientMCTS(MCTS):
    def __init__(self, env, gamma=.99, c=.2, alpha=0.1):
        super(GradientMCTS, self).__init__(env, gamma, c)
        self.gradient = {}
        self.gradient[self.root_node] = [0. for a in env.actions]
        self.alpha = alpha

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
            self.gradient[node] = [0. for a in env.actions]

    def _probability(self, curr_node):
        """Calculate the probability of selecting the given action (of the node)
        using softmax.
        """
        pr = math.e ** self.gradient[curr_node][curr_node.action.value]
        norm_sum = 0
        for child_node in curr_node.parent:
            norm_sum += math.e ** self.gradient[child_node][child_node.action.value]  # noqa: E501
        return pr / norm_sum

    def _update_gradient(self, curr_node, q_val):
        """Update the policy gradient given a node of an action and the current
        Q value.
        """
        self.gradient_updates += 1

        parent_node = curr_node.parent

        mean_q = self.Q[curr_node] / (self.visits[curr_node] + 1)
        delta_q = self.alpha * (q_val - mean_q)

        # Update gradient of selected action (curr_node is the node of the
        # selected action).
        new_gradient = delta_q * (1 - self._probability(curr_node))
        self.gradient[parent_node][curr_node.action.value] += new_gradient

        # Update gradients of all other actions of the parent.
        for child_node in curr_node.parent:
            if child_node == curr_node:
                continue
            new_gradient = delta_q * self._probability(child_node)
            self.gradient[parent_node][child_node.action.value] -= new_gradient

    def backup(self, curr_node, q_val, total_path_len):
        while curr_node is not None:
            discount = self.gamma ** total_path_len
            total_path_len -= 1

            q_val += curr_node.reward * discount

            self.Q[curr_node] += q_val
            self.visits[curr_node] += 1

            if curr_node.action is not None and curr_node.parent is not None:
                self._update_gradient(curr_node, q_val)

            curr_node = curr_node.parent

    def run(self, n_iter=2000, max_actions=100):
        self.gradient_updates = 0  # TODO Remove
        actions = []
        start_time = time.time()
        total_depth = 0
        for j in range(max_actions):
            for i in range(n_iter):
                node, path_len = self.select_expand()
                q_val = self.simulate(node, path_len + total_depth)
                self.backup(node, q_val, path_len + total_depth)
            total_depth += 1

            print(self.gradient[self.root_node])
            print(
                'gradient action',
                self.root_node.children[
                    np.argmax(self.gradient[self.root_node])
                ]
            )

            curr_node = self.root_node.children[
                np.argmax(self.gradient[self.root_node])
            ]

            # print('OG', self._get_best_node(curr_node)[1])

            self.env.step(curr_node.action)
            self.root_node = curr_node
            curr_node.parent = None

            actions.append(curr_node.action)

            if curr_node.is_terminal:
                print('TOTAL GRADIENT UPDATES', self.gradient_updates)
                break
        return actions, time.time() - start_time
