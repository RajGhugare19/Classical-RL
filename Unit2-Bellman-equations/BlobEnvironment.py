import numpy as np
import matplotlib.pyplot as plt
import time
import random

class Blob():
    def __init__(self, SIZE):
        self.size = SIZE
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)

    def __str__(self):
        return f"{self.x}, {self.y}"

class BlobEnvironment():
    def __init__(self):
        self.size = 5
        self.n_actions = 4
        self.player = Blob(self.size)
        self.x = self.player.x
        self.y = self.player.y
        self.color = {"player":(0,0,255)}
        self.reward = 0

    def reset(self):
        self.x = self.player.x
        self.y = self.player.y
        return (self.x, self.y)

    def step(self,action=-1):
        if action == -1:
            print('lolll')
            action = self.player.policy()

        if action == 0:                     #Right
            self.move(x=1, y=0)
        elif action == 1:                   #Down
            self.move(x=0, y=1)
        elif action == 2:                   #Left
            self.move(x=-1, y=0)
        elif action == 3:                   #up
            self.move(x=0, y=-1)
        return self.x,self.y,self.reward

    def move(self, x, y):
        self.reward = 0
        if self.x==1 and self.y==0:
            self.reward = 10
            self.x = 1
            self.y = 4
        elif self.x==3 and self.y==0:
            self.reward = 5
            self.x = 3
            self.y = 2
        else:
            self.x += x
            self.y += y
            if self.x < 0:
                self.x = 0
                self.reward = -1
            elif self.x >= self.size:
                self.x = self.size-1
                self.reward = -1

            if self.y < 0:
                self.y = 0
                self.reward = -1
            elif self.y >= self.size:
                self.y = self.size-1
                self.reward = -1

    def render(self, RenderTime = 100):
        env = np.ones((self.size,self.size,3), dtype = np.uint8)*255
        env[self.y][self.x] = self.color["player"]
        plt.xticks(np.arange(-0.5,4.5,1),np.arange(5))
        plt.yticks(np.arange(-0.5,4.5,1),np.arange(5))
        plt.grid('True')
        plt.imshow(np.array(env))
        plt.pause(RenderTime/100)

    def sample_actions(self):
        return np.random.randint(0, self.n_actions)

    def plot_grid_values(self, values):
        fig, axs = plt.subplots(1,1)
        axs.axis('off')
        the_table = axs.table(cellText=values,bbox=[0, 0, 1, 1],cellLoc="center")
        plt.show()

    def plot_policy(self, policy):
        P = []
        for y in range(5):
            p = []
            for x in range(5):
                if policy[y,x] == 0:
                    p.append("right")
                elif policy[y,x] == 1:
                    p.append("down")
                elif policy[y,x] == 2:
                    p.append("left")
                else:
                    p.append("up")
            P.append(p)
        fig, axs = plt.subplots(1,1)
        axs.axis('off')
        the_table = axs.table(cellText=P,bbox=[0, 0, 1, 1],cellLoc="center")
        plt.show()
