import gym
from REINFORCE import Agent
from utils import plot_score
import numpy as np
import torch
from gym import wrappers


NAME = "LunarLander-v2"
INPUT_DIMS = [8]
GAMMA = 0.99
N_ACTIONS = 4
N_GAMES = 200

if __name__ == '__main__':
    env = gym.make(NAME)
    agent = Agent(lr=0.001, input_dims=INPUT_DIMS, gamma=GAMMA, n_actions=N_ACTIONS,
                    h1=64, h2=32)
    score_history = []
    score = 0
    best_score = -1000

    for i in range(N_GAMES):
        print('episode: ', i, 'score %.3f' % score)
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_rewards(reward)
            state = next_state
            score += reward
        if(np.mean(score_history[-20:])>best_score and i>20):
            torch.save(agent.policy.state_dict(),'./params/'+NAME+'pt.')
            best_score = np.mean(score_history[-20])
        score_history.append(score)
        agent.improve()

    plot_score(score_history,NAME,save=True)
