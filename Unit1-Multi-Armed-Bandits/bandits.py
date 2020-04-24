
#######################################################################
# Copyright (C)                                                       #
# 2020(rajghugare.vnit@gmail.com)                                     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import random


class GaussianStationaryBandit(object):
    def __init__(self, k, mu, sigma):
        self.qstar = np.array(mu)                     #ndarray expected payoff
        self.sigma = np.array(sigma)                  #ndarray standard deviation of payoff
        self.arm_history = np.zeros(k)                #initializing history of all arms taken
        self.regret = []                              #initializing history of regret after every action
        self.best_payoff = np.max(self.qstar)
        self.best_arm = np.argmax(self.qstar)
        self.num_arms = k
        self.rew = 0

    def pull(self, arm):
        self.arm_history[arm] += 1
        self.regret.append(self.best_payoff - self.qstar[arm])
        reward = random.gauss(self.qstar[arm], self.sigma[arm])
        self.reward += reward
        return reward

    def get_ArmHistory(self):
        return self.arm_history

    def get_regret(self):
        return self.regret

    def get_BestArm(self):
        return self.best_arm

    def reset(self):
        self.arm_history = np.zeros(self.num_arms)
        self.regret = []
        self.re = 0

    def get_total_reward(self):
        return self.reward

class BernoulliStationaryBandit(object):
    def __init__(self, k, mu):
        self.qstar = np.array(mu)                     #ndarray expected payoff
        self.arm_history = np.zeros(k)                #initializing history of all arms taken
        self.regret = []                              #initializing history of regret after every action
        self.best_payoff = np.max(self.qstar)
        self.best_arm = np.argmax(self.qstar)
        self.num_arms = k
        self.reward = 0

    def pull(self, arm):
        self.arm_history[arm] += 1
        self.regret.append(self.best_payoff - self.qstar[arm])
        reward =  np.random.choice([1,0], p = [self.qstar[arm], 1-self.qstar[arm]])
        self.reward += reward
        return reward

    def get_ArmHistory(self):
        return self.arm_history

    def get_regret(self):
        return self.regret

    def get_BestArm(self):
        return self.best_arm

    def reset(self):
        self.arm_history = np.zeros(self.num_arms)
        self.regret = []
        self.reward = 0

    def get_total_reward(self):
        return self.reward
