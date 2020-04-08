from Bandits import k_ArmedBandits
from utils import plot_QMaxVsTime
from utils import MeanSquaredError
import numpy as np
import random

k = 5

(k_arms, actions, Qstar) = k_ArmedBandits(k)

#defining a softmax policy
def softmax(Q,beta=1):
    prob = np.copy(np.exp(Q)/np.sum(np.exp(Q)))
    arm = np.random.choice(actions, p = prob)
    return int(arm)


def learn_with_softmax(num_iters,beta=1):
    indicator = np.zeros(k, np.uint8)
    prob = np.zeros(k)
    Q = np.zeros(k)
    Qmax_history = []
    for t in range(1,num_iters+1):
        arm = softmax(Q, beta)
        reward = k_arms[arm].get_reward()
        indicator[arm] += 1
        Q[arm] = (Q[arm]*indicator[arm] + reward)/(indicator[arm]+1)
        Qmax_history.append(np.max(Q))
    return Q,Qmax_history


num_iters = 1000
(Q,Qmax_history) = learn_with_softmax(num_iters)


def experiment(episodes):
    count = 0
    for i in range(episodes):
        (Q,Qmax_history) = learn_with_softmax(num_iters)
        print(i)
        if np.argmax(Q) == np.argmax(Qstar):
            count+=1
    print("variance of all arms is 1")
    print("Expected rewards lie between -10 and 10 for all arms")
    return count/episodes

#uncomment to call the experiment function
#accuracy = experiment(10000)
#print("Accuracy of softmax is >> " , accuracy)

plot_QMaxVsTime(Qmax_history, num_iters, "softmax policy", np.max(Qstar))

print("=========softmax==================")
print("The estimated Q values are >>\n ", Q)
print("The predicted best arm is \n ", np.argmax(Q))

print("The Q* values are >>\n ", Qstar)
print("The real best arm is \n", np.argmax(Qstar))
