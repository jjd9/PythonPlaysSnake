import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import cv2
import sys
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from snakeEnv import GridWorld

# Snake game

env = GridWorld()


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(22, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 4)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.dones = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values



model = Policy()
model.load_state_dict(torch.load("SnakeNet_AC"))
model.eval()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # take most probable action
    action = torch.argmax(probs)

    # the action to take
    return action.item()


def main(render):
    running_reward = 0
    log_interval = 100

    # run inifinitely many episodes
    for i_episode in range(10000):

        # reset environment and episode reward
        env.reset()
        state = env.state
        ep_reward = 0

        board = env.render()
        fps = 10
        size = (board.shape[0],board.shape[1])
        out = cv2.VideoWriter("actor_critic_snake_v1.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        # for each episode, only run 999 steps so that we don't 
        # infinite loop while learning
        done = False
        moves= 0
        while not done:
            moves+=1
        
            # writing to a image array
            out.write(board)

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)

            if render:
                board = env.render()

            ep_reward += reward
            if done:
                break
        cv2.destroyAllWindows()
        out.release()
        if ep_reward > 60:
            break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))



if __name__ == '__main__':
    render = True
    main(render)
    cv2.destroyAllWindows()
