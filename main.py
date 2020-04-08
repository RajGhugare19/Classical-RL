from agents import epsilon_greedy_agent
from agents import softmax_agent
from agents import Median_elimination_agent
from Bandits import GaussianStationaryBandit
from Bandits import BernoulliStationaryBandit
from utils import plot_Regret
from utils import plot_ArmCount
import numpy as np

num_iters = 500

k = 10
mu = np.array([0.11,0.35, 0.57, 0.76, 0.79, 0.84, 0.89, 0.95, 0.955, 0.998])
sigma = [1.5, 3.1, 4.1, 1, 0.1 ,2.1 ,1.1 ,0.61 ,.71 ,1]

#k = 5
#sigma = [1, 1, 1, 1, 1]
#mu = np.array([1,3,6,2,9])

gauss_bandit = GaussianStationaryBandit(k, mu, sigma)
bernoulli_bandit = BernoulliStationaryBandit(k , mu)






if __name__ == "__main__" :

    data = {"bernoulli_bandit":{},"gauss_bandit":{}}

    median_elimination_player = Median_elimination_agent(bernoulli_bandit, epsilon=0.05, delta=0.01)
    data["bernoulli_bandit"]["median_elimination"] = median_elimination_player.play()

    def play_bernoulli():
        epsilon_greedy_player_bernoulli = epsilon_greedy_agent(bernoulli_bandit, 1, num_iters)
        softmax_player_bernoulli = softmax_agent(bernoulli_bandit, 1, num_iters)
        data["bernoulli_bandit"]["epsilon_greedy"] = epsilon_greedy_player_bernoulli.play()
        data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
        plot_Regret(data, 1000, "bernoulli_bandit", k)
        plot_ArmCount(data, 1000, "bernoulli_bandit", k)

    def play_gauss():
        epsilon_greedy_player_gaussian = epsilon_greedy_agent(gauss_bandit, 1, num_iters)
        softmax_player_gaussian = softmax_agent(gauss_bandit, 1, num_iters)
        data["gauss_bandit"]["epsilon_greedy"] = epsilon_greedy_player_gaussian.play()
        data["gauss_bandit"]["softmax"] = epsilon_greedy_player_gaussian.play()
        plot_Regret(data, num_iters, "gauss_bandit", k)
        plot_ArmCount(data, num_iters, "gauss_bandit", k)
