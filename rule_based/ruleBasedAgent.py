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
env.seed(42)
torch.manual_seed(42)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])




def select_action(state):
    xDist = state[0]
    yDist = state[1]
    if xDist < 0:
        print("Left")
        return 0 #left
    elif xDist > 0:
        print("Right")
        return 1 #right
    elif yDist < 0:
        print("Up")
        return 2 #up
    elif yDist > 0:
        print("Down")
        return 3 #down
    else:
        return np.random.randint(0,4)


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
        for t in range(1, 100):

            # select action from policy
            state = env.food - env.snake.body[0,:]
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)

            if render:
                env.render()

            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        cv2.destroyAllWindows()

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > reward_threshold:
            render = True
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    render = True
    main(render)
    cv2.destroyAllWindows()
