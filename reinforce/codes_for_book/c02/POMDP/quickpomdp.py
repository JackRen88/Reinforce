'''
Copyright (c) 2023 by GAC, All Rights Reserved. 
Author: renchengjin
Date: 2023-07-28 16:06:49
LastEditors: JackRen
LastEditTime: 2023-07-28 16:19:23
Description: 
'''


class QuickPOMDP(object):
    def __init__(self, states, 
                       actions, 
                       observations, 
                       initialstate,
                       transition_func,
                       observation_func,
                       reward_func,
                       discount = 0.9,
                       ):
        self.states=states
        self.actions=actions
        self.observations=observations
        self.initialstate=initialstate
        self.discount=discount

        self.transition_func=transition_func
        self.observation_func=observation_func
        self.reward_func=reward_func
