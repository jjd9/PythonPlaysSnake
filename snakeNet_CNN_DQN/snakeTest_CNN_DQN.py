import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import cv2
import sys
import IPython
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from snakeEnv import GridWorld

# Snake game

env = GridWorld(grid_state=True)
env.seed(12)
torch.manual_seed(12)


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(576, 512)

        # Q layer
        self.action_head = nn.Linear(512, 4)


    def forward(self, x):
        x = x/255.0               
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)        
        x = F.leaky_relu(x, 0.1)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.1)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_vals = self.action_head(x)

        return action_vals

model = Policy()
model.load_state_dict(torch.load("SnakeNet_CNN_DQN.pt"))
model.eval()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_values = model(state)

    # take most probable action
    print(action_values)
    action = torch.argmax(action_values)

    # the action to take
    return action.item()


def main(render):
    running_reward = 0
    reward_threshold = 300
    log_interval = 100

    # run inifinitely many episodes
    for i_episode in range(10000):

        # reset environment and episode reward
        env.reset()
        state = env.state
        ep_reward = 0

        # for each episode, only run 999 steps so that we don't 
        # infinite loop while learning
        done = False
        while not done:

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)

            if render:
                env.render()
                time.sleep(0.1)


            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > reward_threshold:
            render = True
            # print("Solved! Running reward is now {} and "
            #       "the last episode runs to {} time steps!".format(running_reward, t))
            # break


if __name__ == '__main__':
    render = True
    main(render)
    cv2.destroyAllWindows()
