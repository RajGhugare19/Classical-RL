from agents import epsilon_greedy_agent
from agents import softmax_agent
from agents import Median_elimination_agent
from agents import UCB
from Bandits import GaussianStationaryBandit
from Bandits import BernoulliStationaryBandit
from utils import plot_ArmCount
from utils import plot_regret
import numpy as np


#gauss_bandit = GaussianStationaryBandit(k, mu, sigma)
#sigma = [1.5, 3.1, 4.1, 1, 0.1 ,2.1 ,1.1 ,0.61 ,.71 ,1]
#epsilon_greedy_player_gaussian = epsilon_greedy_agent(gauss_bandit, 1, num_iters)
#softmax_player_gaussian = softmax_agent(gauss_bandit, 1, num_iters)


num_iters = 5000

k = 10
mu = np.array([0.1,0.5,0.7,0.73,0.756,0.789,0.81,0.83,0.855,0.865])
mu = np.arange(10)*0.1
bernoulli_bandit = BernoulliStationaryBandit(k , mu)

#initializing all the players
epsilon_greedy_player_bernoulli = epsilon_greedy_agent(bernoulli_bandit, 1, num_iters)
softmax_player_bernoulli = softmax_agent(bernoulli_bandit, 0.1, num_iters)
median_elimination_player = Median_elimination_agent(bernoulli_bandit, epsilon=0.1, delta=0.1)
UCB_player = UCB(bernoulli_bandit,num_iters)

def play_UCB():
    data["bernoulli_bandit"]["UCB"] = UCB_player.play()
    data["bernoulli_bandit"]["UCB"] = UCB_player.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "UCB", k)
    plot_ArmCount(data, num_iters, "bernoulli_bandit", "UCB", k)


def play_median_elimination():
    data["bernoulli_bandit"]["median_elimination"] = median_elimination_player.play()


def play_epsilon_greedy():
    data["bernoulli_bandit"]["epsilon_greedy"] = epsilon_greedy_player_bernoulli.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "epsilon_greedy", k)
    plot_ArmCount(data, num_iters, "bernoulli_bandit", "epsilon_greedy", k)

def play_softmax():
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "softmax", k)
    plot_ArmCount(data, num_iters, "bernoulli_bandit", "softmax", k)







if __name__ == "__main__" :

    data = {"bernoulli_bandit":{},"gauss_bandit":{}}
    data["bernoulli_bandit"]["UCB"] = UCB_player.play()
    data["bernoulli_bandit"]["median_elimination"] = median_elimination_player.play()
    data["bernoulli_bandit"]["epsilon_greedy"] = epsilon_greedy_player_bernoulli.play()
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    plot_ArmCount(data, num_iters, "bernoulli_bandit", k)
