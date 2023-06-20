from random import random, choice
from agents import SarsaAgent, QAgent
from gridworld import CliffWalk, CliffWalk2
from gym import Env
import gym

from utils import greedy_policy, learning_curve, str_key

env = CliffWalk()
env.reset()
env.render()
