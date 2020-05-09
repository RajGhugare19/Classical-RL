import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot(target_policy):
    usable = np.zeros([11,10])
    non_usable = np.zeros([11,10])
    for i in target_policy:
        if i[0]>10:
            if i[2]:
                usable[i[0]-11,i[1]-1]=target_policy[i]
            else:
                non_usable[i[0]-11,i[1]-1]=target_policy[i]
    usable = np.flip(usable,0)
    non_usable = np.flip(non_usable,0)
    ax = sns.heatmap(usable, linewidth=0, cbar=False)
    plt.xlabel('dealer showing')
    plt.title("Non Usable ace")
    plt.title("Usable ace [Black=Stick,Off-white=Hit]")
    plt.yticks(np.arange(11),['21','20','19','18','17','16','15','14','13','12','11'])
    plt.xticks(np.arange(0,10)+0.5,['1','2','3','4','5','6','7','8','9','10'])
    ax.yaxis.tick_right()
    plt.show()
    ax = sns.heatmap(non_usable, linewidth=0, cbar=False)
    plt.ylabel('player sum')
    plt.xlabel('dealer showing')
    plt.title("Non Usable ace [[Black=Stick,Off-white=Hit]]")
    plt.yticks(np.arange(11),['21','20','19','18','17','16','15','14','13','12','11'])
    plt.xticks(np.arange(0,10)+0.5,['1','2','3','4','5','6','7','8','9','10'])
    ax.yaxis.tick_right()
    plt.show()
