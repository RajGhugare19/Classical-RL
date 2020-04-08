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

    def pull(self, arm):
        self.arm_history[arm] += 1
        self.regret.append(self.best_payoff - self.qstar[arm])
        return random.gauss(self.qstar[arm], self.sigma[arm])

    def get_ArmHistory(self):
        return self.arm_history

    def get_regret(self):
        return self.regret

    def get_BestArm(self):
        return self.best_arm

    def reset(self):
        self.arm_history = np.zeros(self.num_arms)
        self.regret = []

class BernoulliStationaryBandit(object):
    def __init__(self, k, mu):
        self.qstar = np.array(mu)                     #ndarray expected payoff
        self.arm_history = np.zeros(k)                #initializing history of all arms taken
        self.regret = []                              #initializing history of regret after every action
        self.best_payoff = np.max(self.qstar)
        self.best_arm = np.argmax(self.qstar)
        self.num_arms = k

    def pull(self, arm):
        self.arm_history[arm] += 1
        self.regret.append(self.best_payoff - self.qstar[arm])
        return np.random.choice([1,0], p = [self.qstar[arm], 1-self.qstar[arm]])

    def get_ArmHistory(self):
        return self.arm_history

    def get_regret(self):
        return self.regret

    def get_BestArm(self):
        return self.best_arm

    def reset(self):
        self.arm_history = np.zeros(self.num_arms)
        self.regret = []
