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


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1024, 512)

        # Actor layer
        self.action_head = nn.Linear(512, 4)

        # Critic layer
        self.critic_head = nn.Linear(512, 1)


    def forward(self, x):
        x = x/255.0               
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)        
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_probs = F.softmax(self.action_head(x), dim=-1)

        # critic, judges current state value
        state_val = self.critic_head(x)

        return action_probs, state_val

model = Policy()

def select_action(state):
    with torch.no_grad():
        state = torch.from_numpy(state.copy()).float().unsqueeze(0)
        action_probs, _ = model(state)
        print(action_probs)
        action = torch.argmax(action_probs).item()
        reverse_list = [1, 0, 3, 2]
        step = env.snake.action_map[action]
        if np.all(env.snake.body[1,:]==env.snake.body[0,:]+step):
            action = reverse_list[action]
        return action

model = Policy()
model.load_state_dict(torch.load("SnakeNet_CNN_AC"))
model.eval()


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
