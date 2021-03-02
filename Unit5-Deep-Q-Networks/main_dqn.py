import gym
from dqn import cart_agent
import numpy as np
import torch
from utils import plot_durations
from utils import plot_learning_curve

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    A =  agent(epsilon=1,eps_decay=0.005,epsilon_min=0.01,gamma=0.99,l_r=0.0001,n_actions=3,
                memory=20000,batch_size=32,target_update=7,env=env,save=True)

    scores, avg_score, epsilon_history = [], [], []
    best_score = -np.inf
    n_games = 1000
    score = 0

    print("Save is currently !!!!!!!!!!!!!!!!!! ", A.save)

    for i in range(n_games):
        A.env.reset()
        last_screen = A.get_state()
        current_screen = A.get_state()
        state = current_screen-last_screen

        done = False
        score = 0

        if i%20==0 and i>0:
            plot_durations(scores, 0.001)
            print('----------------- training --------------------')
            print('epsiode number', i)
            print("Average score ",avg_score[-1])
            print('----------------- training --------------------')

        while not done:
            action = A.choose_action(state)

            _, reward, done, _ = A.env.step(action)

            last_screen = current_screen
            current_screen = A.get_state()

            next_state = current_screen - last_screen

            A.store_experience(state,action,reward,done,next_state)
            A.learn_with_experience_replay()

            score += reward
            state = next_state

        scores.append(score)
        if i>30:
            avg_score.append(np.mean(scores[-30:]))
        else:
            avg_score.append(np.mean(scores))

        if avg_score[-1] > best_score:
            torch.save(A.policy_net.state_dict(),'/home/raj/My_projects/DQN/MountanCar.pt')
            best_score = avg_score[-1]
            print("***************\ncurrent best average score is "+ str(best_score) +"\n***************")

        if i%A.target_update == 0:
            A.target_net.load_state_dict(A.policy_net.state_dict())

        A.epsilon_decay()

    plot_durations(scores,5)
