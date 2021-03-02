import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch import optim
import torchvision.transforms as T
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepQNetwork(nn.Module):

    def __init__(self,learning_rate,h,w,n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size(size, kernel_size = 3, stride = 1):
            return (size - kernel_size)// stride  + 1

        convw = conv2d_size(conv2d_size(conv2d_size(w,8,4),4,2))
        convh = conv2d_size(conv2d_size(conv2d_size(h,8,4),4,2))
        lin_1 = convw*convh


        self.linear1 = nn.Linear(lin_1*64,256)
        self.linear2 = nn.Linear(256, n_actions)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = torch.relu(self.linear1(x))
        action_values = self.linear2(x)
        return action_values

class agent():

    def __init__(self,epsilon,eps_decay,epsilon_min,gamma,l_r,n_actions,memory,batch_size,target_update,env,save=False):
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.env = env
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = memory
        self.memory_count = 0
        self.ROWS = 84
        self.COLS = 84
        self.state_memory = torch.zeros([self.memory,1,self.ROWS,self.COLS],dtype = torch.float32)
        self.next_state_memory = torch.zeros([self.memory,1,self.ROWS,self.COLS],dtype = torch.float32)
        self.action_memory = torch.zeros(self.memory,dtype=torch.int32)
        self.terminal_memory = torch.zeros(self.memory,dtype=torch.uint8)
        self.reward_memory = torch.zeros(self.memory)
        self.policy_net = DeepQNetwork(learning_rate = l_r,h=self.ROWS,w=self.COLS,n_actions=self.n_actions).to(device)
        self.target_net = DeepQNetwork(learning_rate = l_r,h=self.ROWS,w=self.COLS,n_actions=self.n_actions).to(device)
        self.target_update = target_update
        self.save = save



    def choose_action(self,state):
        r = np.random.random()
        if r<self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_val = self.policy_net.forward(state)
                action = torch.argmax(q_val).item()

        return action


    def store_experience(self,state,action,reward,terminal,next_state):
        index = self.memory_count%self.memory

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1-terminal
        self.next_state_memory[index] = next_state

        self.memory_count+=1

    def get_state(self):

        screen = self.env.render(mode='rgb_array')
        screen_1 = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        r_screen = cv2.resize(screen_1, (self.ROWS,self.COLS), interpolation=cv2.INTER_AREA)
        r_screen = np.array(r_screen)
        r_screen = np.expand_dims(r_screen,axis=0)
        r_screen = torch.Tensor(r_screen)
        return r_screen.unsqueeze(0).to(device)


    def learn_with_experience_replay(self):
        if self.memory_count < self.batch_size:
            return

        if self.memory_count < self.memory:
            mem = self.memory_count
        else:
            mem = self.memory

        self.policy_net.optimizer.zero_grad()


        batch = np.random.choice(mem, self.batch_size, replace=False)

        state_batch = self.state_memory[batch].to(device)
        action_batch = self.action_memory[batch]
        new_state_batch = self.next_state_memory[batch].to(device)
        reward_batch = self.reward_memory[batch].to(device)
        terminal_batch = self.terminal_memory[batch].to(device)


        q_val = self.policy_net.forward(state_batch).to(device)

        q_next = self.target_net.forward(new_state_batch).to(device).detach()

        q_target = self.policy_net.forward(state_batch).to(device).detach()

        batch_index = np.arange(self.batch_size)
        action_values = torch.max(q_next,1)[0]

        q_target[batch_index, np.array(action_batch)] = reward_batch + self.gamma*action_values*terminal_batch

        loss = self.policy_net.criterion(q_val,q_target).to(device)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()



    def epsilon_decay(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon = self.epsilon-self.eps_decay
        return self.epsilon


print('Done')
