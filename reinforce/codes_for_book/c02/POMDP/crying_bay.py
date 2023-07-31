'''
Copyright (c) 2023 by GAC, All Rights Reserved. 
Author: renchengjin
Date: 2023-07-28 16:04:34
LastEditors: JackRen88
LastEditTime: 2023-07-30 09:09:17
Description: 
'''
import quickpomdp

states = ["hungry", "full"]
actions = ["feed", "ingore"]
observation = ["crying", "quiet"]


def transition(s, a, s1):
    if a == "feed":
        if s1 == "hungry":
            return 0.0
        elif s1 == "full":
            return 1.0
    elif s == "hungry" and a == "ingore":
        if s1 == "hungry":
            return 1.0
        elif s1 == "full":
            return 0.0
    elif s == "full" and a == "ingore":
        if s1 == "hungry":
            return 0.1
        elif s1 == "full":
            return 0.9


def reward(s, a):
    return (-10 if s == "hungry" else 0) + (-5 if a == "feed" else 0)


def observation(s, a, s1, o1):
    if s1 == "hungry":
        if o1 == "crying":
            return 0.8
        elif o1 == "quiet":
            return 0.2
    elif s1 == "full":
        if o1 == "crying":
            return 0.1
        elif o1 == "quiet":
            return 0.9


if __name__ == "__main__":
    t = transition("full", "feed", "feed")
    s = "hungry"
    a = "feed"
    r = reward(s, a)
    print("test reward function,r= {}".format(r))
    pomdp = QuickPOMDP()
