from dqn import cart_agent
import torch
import gym
import time
import numpy as np
from utils import plot_learning_curve

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
env = gym.make('MountainCar-v0').unwrapped

if __name__ == '__main__':
    player =  agent(epsilon=0,eps_decay=0,epsilon_min=0,gamma=0,l_r=0,n_actions=3,
                memory=0,batch_size=0,target_update=0,env = env,save = True)
    n_games = 3
    scores = []
    player.policy_net.load_state_dict(torch.load('/home/raj/My_projects/DQN/MountanCar.pt'))

    for i in range(n_games):
        env.reset()
        last_screen = player.get_state()
        current_screen = player.get_state()
        state = current_screen-last_screen

        done = False
        score = 0
        while not done:
            action = player.choose_action(state)
            time.sleep(0.05)
            _, reward, done, _ = player.env.step(action)

            last_screen = current_screen
            current_screen = player.get_state()

            next_state = current_screen - last_screen
            score += reward
            state = next_state


        scores.append(score)
    print(np.mean(scores))
    plot_learning_curve(i, scores,0)
