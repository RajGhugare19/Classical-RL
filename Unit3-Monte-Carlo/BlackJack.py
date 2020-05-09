#######################################################################
# Copyright (C)                                                       #
# 2020(rajghugare.vnit@gmail.com)                                     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import plot

env= gym.make('Blackjack-v0')
GAMMA = 1

playerSum = list(np.arange(4,22))
agentCard = list(np.arange(1,11))
playerAce = [False,True]
actionSpace = [0,1]
stateSpace = []

target_policy = {}
Q = {}
C = {}
for p in playerSum:
    for a in agentCard:
        for ace in playerAce:
            stateSpace.append((p,a,ace))
            m = -1
            for action in actionSpace:
                Q[(p,a,ace),action] = random.random()
                if Q[(p,a,ace),action] > m:
                    m = Q[(p,a,ace),action]
                    argmax = action
                C[(p,a,ace),action] = 0
            target_policy[(p,a,ace)] = argmax

def behaviour_policy():
    r = random.uniform(0,1)
    if r<0.5:
        return 0
    else:
        return 1


# Monte Carlo Off policy control to find optimal policy
for i in range(1000000):
    states = []
    actions = []
    rewards = []
    done = False
    states.append(env.reset())
    a = behaviour_policy()
    actions.append(a)
    while True:
        (s,r,done,_) = env.step(a)
        rewards.append(r)
        if done:
            break
        a = behaviour_policy()
        actions.append(a)
        states.append(s)
    G = 0
    W = 1      #Importance sampling ratio
    for i in range(len(states)):
        G = G + GAMMA*rewards[-1-i]
        C[states[-i-1],actions[-i-1]] += W
        Q[states[-i-1],actions[-i-1]] = Q[states[-i-1],actions[-i-1]] + W*(G-Q[states[-i-1],actions[-i-1]])/C[states[-i-1],actions[-i-1]]
        m = -1
        for action in actionSpace:
            if Q[states[-1-i],action] > m:
                m = Q[states[-1-i],action]
                argmax = action
        target_policy[states[-1-i]] = argmax
        if actions[-i-1] != argmax:
            break
        W = W*(1/0.5)

def play(n):
    win = 0
    loss = 0
    draw = 0
    for i in range(n):
        score = 0
        done = False
        s = env.reset()
        while not done:
            a = target_policy[s]
            (s,r,done,_) = env.step(a)
            score += r
        if score==0:
            draw += 1
        elif score==1:
            win += 1
        else:
            loss +=1
    print(win)
    print(loss)
    print(draw)

plot(target_policy)
