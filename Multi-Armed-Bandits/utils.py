import numpy as np
import matplotlib.pyplot as plt

def MeanSquaredError(q,qstar):
    return (np.sum((q-qstar)**2)/np.max(qstar))

def plot_Regret(data, num_iters, bandit, k):
    (e_Q, e_regret, e_arm_history) = data[bandit]["epsilon_greedy"]
    (s_Q, s_regret, s_arm_history) = data[bandit]["softmax"]
    t = np.arange(num_iters)
    fig,a = plt.subplots(2,1)
    a[0].plot(t, e_regret, color="red", label="EpsilonGreedy")
    a[1].plot(t, s_regret, color="green", label="Softmax")
    a[0].set_xlabel("Number of trials")
    a[0].set_ylabel("Regret")
    a[1].set_xlabel("Number of trials")
    a[1].set_ylabel("Regret")
    a[0].legend()
    a[1].legend()
    plt.tight_layout()
    plt.show()


def plot_ArmCount(data, num_iters, bandit, k):
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

def plot_regret(data, num_iters, bandit, player, k):
    x_index = []
    for i in range(k):
        x_index.append(str(i))
    (_, regret, arm_history) = data[bandit][player]
    t = np.arange(num_iters)
    plt.plot(t, regret, color='green', label=player)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
