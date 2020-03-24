import cProfile
import gym
import gym_minigrid  # noqa F401
from mcts import MCTS

env = gym.make('MiniGrid-Empty-8x8-v0')
mcts_obj = MCTS(env)


def run():
    mcts_obj.run()


cProfile.runctx("run()", globals(), locals(), "4.profile")
