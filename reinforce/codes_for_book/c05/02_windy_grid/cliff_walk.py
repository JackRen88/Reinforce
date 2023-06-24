from random import random, choice
from agents import SarsaAgent, QAgent
from gridworld import CliffWalk, CliffWalk2
from gym import Env
import gym

from utils import greedy_policy, learning_curve, str_key

env = CliffWalk()
env.reset()
env.render()


q_agent = QAgent(env, capacity = 10000)
sarsa_agent = SarsaAgent(env, capacity=10000)

sarsa_sta = sarsa_agent.learning(display=False,
                                 max_episode_num=10000,
                                 epsilon=0.1,
                                 decaying_epsilon=False)
q_sta = q_agent.learning(display = False,
                         max_episode_num = 10000,
                         epsilon = 0.1, 
                         decaying_epsilon=False)
