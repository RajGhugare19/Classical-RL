
#######################################################################
# Copyright (C)                                                       #
# 2020(rajghugare.vnit@gmail.com)                                     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from BlobEnvironment import BlobEnvironment

env = BlobEnvironment()

EPSILON = 0.01  #this is an optimality factor
GAMMA = 0.9    #As defined in the problem itself

def value_iteration():
    n = 0
    v = np.zeros([5,5])                         #This should still be considered as a 25 dimensional vector
    v_new = np.zeros([5,5])                     #It is in the form of 5 by 5 matrix for better visual undertanding and slightly easier implementation
    while True:
        for y in range(5):
            for x in range(5):
                v_temp = np.zeros(4)
                for action in range(4):
                    env.x = x
                    env.y = y
                    x_next,y_next,reward = env.step(action)
                    v_temp[action] = reward + GAMMA*v[y_next,x_next]
                v_new[y,x] = np.max(v_temp)
        if np.max(np.abs(v - v_new)) < EPSILON*(1-GAMMA)/(2*GAMMA):
            env.plot_grid_values(np.round(v_new,decimals=2))
            break
        v = np.copy(v_new)

def policy_iteration():
    n = 0
    policy = np.zeros([5,5],dtype = np.uint8)
    v = np.zeros([5,5])
    v_new = np.zeros([5,5])
    while True:
        #Policy evaluation
        while True:
            for y in range(5):
                for x in range(5):
                    action = policy[y,x]
                    env.x = x
                    env.y = y
                    x_next,y_next,reward = env.step(action)
                    v_new[y,x] = reward + GAMMA*v[y_next,x_next]
            if np.max(np.abs(v - v_new)) < EPSILON*(1-GAMMA)/(2*GAMMA):
                break
            v = np.copy(v_new)
        #Policy improvement
        new_policy = np.zeros([5,5],dtype=np.uint8)
        for y in range(5):
            for x in range(5):
                v_temp = np.zeros(4)
                for action in range(4):
                    env.x = x
                    env.y = y
                    x_next,y_next,reward = env.step(action)
                    v_temp[action] = reward + GAMMA*v[y_next,x_next]
                new_policy[y,x] = np.argmax(v_temp)
        if np.array_equal(policy,new_policy):
            break
        policy = np.copy(new_policy)
    env.plot_policy(policy)
    return policy


def Play_optimally(policy):
    (x,y) = env.reset()
    for i in range(25):
        action = policy[y,x]
        (x,y,reward) = env.step(action)
        print(reward)
        env.render()


value_iteration()

policy = policy_iteration()
Play_optimally(policy)
