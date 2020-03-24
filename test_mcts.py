import unittest
from mcts import MCTS

import gym
import gym_minigrid  # noqa F401


class TestMCTS(unittest.TestCase):
    def test_select_expand(self):
        env = gym.make('MiniGrid-Empty-5x5-v0')
        mcts_obj = MCTS(env)
        self.assertEqual(mcts_obj.root_node.children, [])
        path = mcts_obj.select_expand()
        self.assertEqual(path, [0])
        self.assertEqual(len(mcts_obj.root_node.children), 7)
