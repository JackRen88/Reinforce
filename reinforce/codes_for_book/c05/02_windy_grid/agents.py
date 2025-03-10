#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# agents of reinforcment learning

# Author: Qiang Ye
# Date: July 27, 2017

from random import random, choice
from gym import Env
import gym
from gridworld import *
from core import Transition, Experience, Agent
from utils import str_key, set_dict, get_dict
from utils import epsilon_greedy_pi, epsilon_greedy_policy
from utils import greedy_policy, learning_curve
# from approximator import Approximator
# import torch


class SarsaAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(SarsaAgent, self).__init__(env, capacity)
        self.Q = {}
        self.name ="SarsaAgent"
    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False, lambda_ = None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, self.Q, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, self.Q, epsilon)
            old_q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * q_prime
            #alpha = alpha / num_episode
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.name)
            print(self.experience.last_episode)
        return time_in_episode, total_reward
    

    
class SarsaLambdaAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(SarsaLambdaAgent, self).__init__(env, capacity)
        self.Q = {}

    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, lambda_ = 0.9, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        E = {}
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            
            q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            delta = r1 + gamma * q_prime - q
            
            e = get_dict(E, s0, a0)
            e += 1
            set_dict(E, e, s0, a0)

            for s in self.S:
                for a in self.A:
                    e_value = get_dict(E, s, a)
                    old_q = get_dict(self.Q, s, a)
                    new_q = old_q + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    set_dict(self.Q, new_q, s, a)
                    set_dict(E, new_e, s, a)
                    
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward    
            
        
class QAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(QAgent, self).__init__(env, capacity)
        self.Q = {}
        self.name ="QAgent"
    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)
    
    def learning_method(self, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False, lambda_ = None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        # a0 = self.perform_policy(s0, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            a0 = self.perform_policy(s0, self.Q, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            # self.policy = greedy_policy
            a1 = greedy_policy(self.A, s1, self.Q)
            old_q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * q_prime
            #alpha = alpha / num_episode
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)
            # s0, a0 = s1, a1
            s0 = s1
            time_in_episode += 1
        if display:
            print(self.name)
            print(self.experience.last_episode)
        return time_in_episode, total_reward    
            
# class ApproxQAgent(Agent):
#     '''使用近似的价值函数实现的Q学习个体
#     '''
#     def __init__(self, env: Env = None,
#                        trans_capacity = 20000,
#                        hidden_dim: int = 16):
#         if env is None:
#             raise "agent should have an environment"
#         super(ApproxQAgent, self).__init__(env, trans_capacity)
#         self.input_dim, self.output_dim = 1, 1
#         if isinstance(env.observation_space, spaces.Discrete):
#             self.input_dim = 1
#         elif isinstance(env.observation_space, spaces.Box):
#             self.input_dim = env.observation_space.shape[0]

#         if isinstance(env.action_space, spaces.Discrete):
#             self.output_dim = env.action_space.n
#         elif isinstance(env.action_space, spaces.Box):
#             self.output_dim = env.action_space.shape[0]

#         # print("{},{}".format(self.input_dim, self.output_dim))
#         self.hidden_dim = hidden_dim
#         self.Q = Approximator(dim_input = self.input_dim,
#                               dim_output = self.output_dim,
#                               dim_hidden = self.hidden_dim)
#         self.PQ = self.Q.clone() # 更新参数的网络
#         return

#     def _decayed_epsilon(self,cur_episode: int, 
#                               min_epsilon: float, 
#                               max_epsilon: float, 
#                               target_episode: int) -> float:
#         '''获得一个在一定范围内的epsilon
#         '''
#         slope = (min_epsilon - max_epsilon) / (target_episode)
#         intercept = max_epsilon
#         return max(min_epsilon, slope * cur_episode + intercept)

#     def _curPolicy(self, s, epsilon = None):
#         '''依据更新策略的价值函数(网络)产生一个行为
#         '''
#         Q_s = self.PQ(s)
#         rand_value = random()
#         if epsilon is not None and rand_value < epsilon:
#             return self.env.action_space.sample()
#         else:
#             return int(np.argmax(Q_s))
        
#     def performPolicy(self, s, epsilon = None):
#         return self._curPolicy(s, epsilon)


#     def _update_Q_net(self):
#         '''将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
#         '''
#         self.Q = self.PQ.clone()

    
#     def _learn_from_memory(self, gamma, batch_size, learning_rate, epochs):
#         trans_pieces = self.sample(batch_size)  # 随机获取记忆里的Transmition
#         states_0 = np.vstack([x.s0 for x in trans_pieces])
#         actions_0 = np.array([x.a0 for x in trans_pieces])
#         reward_1 = np.array([x.reward for x in trans_pieces])
#         is_done = np.array([x.is_done for x in trans_pieces])
#         states_1 = np.vstack([x.s1 for x in trans_pieces])

#         X_batch = states_0
#         y_batch = self.Q(states_0)  # 得到numpy格式的结果

#         Q_target = reward_1 + gamma * np.max(self.Q(states_1), axis=1)*\
#             (~ is_done) # is_done则Q_target==reward_1
#         y_batch[np.arange(len(X_batch)), actions_0] = Q_target
#         # loss is a torch Variable with size of 1
#         loss = self.PQ.fit(x = X_batch, 
#                            y = y_batch, 
#                            learning_rate = learning_rate,
#                            epochs = epochs)

#         mean_loss = loss.sum().data[0] / batch_size
#         self._update_Q_net()
#         return mean_loss

#     def learning(self, gamma = 0.99,
#                        learning_rate=1e-5, 
#                        max_episodes=1000, 
#                        batch_size = 64,
#                        min_epsilon = 0.2,
#                        epsilon_factor = 0.1,
#                        epochs = 1):

#         total_steps, step_in_episode, num_episode = 0, 0, 0
#         target_episode = max_episodes * epsilon_factor
#         while num_episode < max_episodes:
#             epsilon = self._decayed_epsilon(cur_episode = num_episode,
#                                             min_epsilon = min_epsilon, 
#                                             max_epsilon = 1,
#                                             target_episode = target_episode)
#             self.state = self.env.reset()
#             # self.env.render()
#             step_in_episode = 0
#             loss, mean_loss = 0.00, 0.00
#             is_done = False
#             while not is_done:
#                 s0 = self.state

#                 a0  = self.performPolicy(s0, epsilon)
#                 s1, r1, is_done, info, total_reward = self.act(a0)
#                 # self.env.render()
#                 step_in_episode += 1
                
#                 if self.total_trans > batch_size:
#                     loss += self._learn_from_memory(gamma, 
#                                                     batch_size, 
#                                                     learning_rate,
#                                                     epochs)
#             mean_loss = loss / step_in_episode
#             print("{0} epsilon:{1:3.2f}, loss:{2:.3f}".
#                 format(self.experience.last, epsilon, mean_loss))
#             # print(self.experience)
#             total_steps += step_in_episode
#             num_episode += 1

#         return   


# def testApproxQAgent():
#     env = gym.make("PuckWorld-v0")
#     #env = SimpleGridWorld()
#     directory = "/home/qiang/workspace/reinforce/python/monitor"
    
#     env = gym.wrappers.Monitor(env, directory, force=True)
#     agent = ApproxQAgent(env,
#                          trans_capacity = 50000, 
#                          hidden_dim = 32)
#     env.reset()
#     print("Learning...")  
#     agent.learning(gamma=0.99, 
#                    learning_rate = 1e-3,
#                    batch_size = 64,
#                    max_episodes=5000,   # 最大训练Episode数量
#                    min_epsilon = 0.2,   # 最小Epsilon
#                    epsilon_factor = 0.3,# 开始使用最小Epsilon时Episode的序号占最大
#                                         # Episodes序号之比，该比值越小，表示使用
#                                         # min_epsilon的episode越多
#                     epochs = 2          # 每个batch_size训练的次数
#                    )


# if __name__ == "__main__":
#     testApproxQAgent()