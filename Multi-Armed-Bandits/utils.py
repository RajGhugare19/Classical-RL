import numpy as np
import matplotlib.pyplot as plt

def epsilon_greedy_softmax_compare(data, num_iters, bandit, k):
    #Compares arm historyt of espilon greedy and softmax
    x_index = []
    for i in range(k):
        x_index.append(str(i))
    (e_Q, e_regret, e_arm_history) = data[bandit]["epsilon_greedy"]
    (s_Q, s_regret, s_arm_history) = data[bandit]["softmax"]

    location = np.arange(k)
    width = 0.35

    (fig, ax) = plt.subplots(1,1)

    rects1 = ax.bar(location-width/2, e_arm_history, width=width, color='red', label='epsilon_greedy', edgecolor='black')
    rects2 = ax.bar(location+width/2, s_arm_history, width=width, color='green', label='softmax', edgecolor='black')

    ax.set_ylabel('Arm pull history')
    ax.set_title('Number of times arm pulled')
    ax.set_xticks(location)
    ax.set_xticklabels(x_index)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_ArmCount(data, num_iters, bandit, k):
    x_index = []
    for i in range(k):
        x_index.append(str(i))
    location = np.arange(k)
    (e_Q, e_regret, e_arm_history, _) = data[bandit]["epsilon_greedy"]
    (s_Q, s_regret, s_arm_history, _) = data[bandit]["softmax"]
    (u_Q, u_regret, u_arm_history, _) = data[bandit]["UCB"]
    (fig, ax) = plt.subplots(1,1)
    bar1 = ax.bar(location, e_arm_history, label="epsilon_greedy", fill=False, edgecolor='green')
    bar2 = ax.bar(location, s_arm_history, label="softmax", fill=False, edgecolor='red')
    bar3 = ax.bar(location, u_arm_history, label="UCB1", fill=False, edgecolor='purple')
    ax.set_ylabel('Arm pull history')
    ax.set_title('Number of times arm pulled')
    ax.set_xticks(location)
    ax.set_xticklabels(x_index)
    ax.legend()
    fig.tight_layout()
    plt.show()



def plot_regret(data, num_iters, bandit, player, k):
    #Plots regret of any one bandit at a time
    (Q, regret, arm_history, _) = data[bandit][player]
    t = np.arange(num_iters)
    plt.plot(t, regret, color='green', label=player)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
