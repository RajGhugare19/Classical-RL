import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import matplotlib
import torch
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_learning_curve(episode, scores, epsilon):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('episode %s. average_reward: %s' % (episode, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.subplot(132)
    plt.title('epsilon')
    plt.plot(epsilon)
    plt.show()

def plot_playing_curve(episode, scores):
    clear_output(True)
    plt.figure(figsize=(5,5))
    plt.title('episode %s. average_reward: %s' % (episode, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.show()

def plot_durations(scores,pause):
    plt.ion()
    plt.figure(2)
    plt.clf()

    durations_t = torch.tensor(scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Scores')
    plt.plot(durations_t.numpy())
    # Take 20 episode averages and plot them too
    if len(durations_t) >= 20:
        means = durations_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy())

    plt.pause(pause)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
