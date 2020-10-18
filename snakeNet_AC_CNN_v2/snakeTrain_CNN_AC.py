import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import cv2
import sys
import IPython
import matplotlib.pyplot as plt
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from snakeEnv import GridWorld

from collections import deque
import random

import IPython, sys

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/snakeNet_CNN_AC')


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state      = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward,next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


# Snake game

env = GridWorld(grid_state=True)
env.seed(42)
torch.manual_seed(42)

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

device = torch.device("cpu")#"cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on: ", device)

save_net_path = "SnakeNet_CNN_AC"
# save_net_path = "drive/My Drive/SnakeNet_CNN_AC"

model = Policy()
if os.path.exists(save_net_path):
  model.load_state_dict(torch.load(save_net_path))
  model.eval()
model.to(device)

temp_optimizer = optim.Adam(model.parameters(), lr=7e-4)

eps = np.finfo(np.float32).eps.item()

def select_action(state, use_rules=False):
    with torch.no_grad():
        if use_rules:
            xDist, yDist = env.food - env.snake.body[0,:]
            if xDist < 0:
                return 0 #left
            elif xDist > 0:
                return 1 #right
            elif yDist < 0:
                return 2 #up
            elif yDist > 0:
                return 3 #down
            else:
                return np.random.randint(0,4)
        else:
            state = torch.from_numpy(state.copy()).float().unsqueeze(0)
            action_probs, _ = model(state)
            m = Categorical(action_probs)
            action = m.sample().item()
            reverse_list = [1, 0, 3, 2]
            step = env.snake.action_map[action]
            if np.all(env.snake.body[1,:]==env.snake.body[0,:]+step):
                action = reverse_list[action]
            return action            



def update_model(batch):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """

    state, action, returns, next_state, done = batch

    state      = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action     = torch.LongTensor(action).reshape(-1,1).to(device)
    returns     = torch.FloatTensor(returns).reshape(-1,1).to(device)
    done       = torch.FloatTensor(done).reshape(-1,1).to(device)

    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    probs, values = model(state)

    # critic loss
    value_loss = (returns - values).pow(2).mean()

    # actor loss
    action_mask = torch.zeros((len(action),4)).float()
    for i in range(len(action)):
        action_mask[i,action[i]] = 1.0
    select_probs = (probs*action_mask).sum(1).reshape(-1,1)

    advantage = torch.clamp(returns - values.detach(),0, np.inf)
    policy_loss = (-torch.log(select_probs)*advantage).mean()
    entropy_loss = (-(torch.log(probs)*probs).sum(1)).mean()

    loss = value_loss + policy_loss - 0.01*entropy_loss

    # perform backprop
    temp_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    temp_optimizer.step()

    # reset rewards and action buffer
    loss_val = loss.item()

    return loss_val


def main():
    render = False
    log_interval = 10

    running_reward_rule = 0    
    running_reward_net = 0    

    loss_val = 0.0
    
    # setup replay buffer
    batch_size = 100
    max_buffer_size = 100000
    initial_buffer_size = 10000
    replay_buffer = ReplayBuffer(max_buffer_size)

    # used for frame multiplication
    # rot = 1
    act_90 = [3, 2, 0, 1]
    # rot = 2
    act_180 = [1, 0, 3, 2]
    # rot = 3
    act_270 = [2, 3, 1, 0]
    # flip
    act_flip = [1, 0, 2, 3]
    
    rotate_action_list = [act_90, act_180, act_270]

    num_frames = 0

    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset(radius=np.random.randint(1,20))
        ep_reward_rule = 0
        ep_reward_net = 0

        temp_buffer = []

        use_rules = np.random.choice([True,False], p=[0.1,0.9])
        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state, use_rules=use_rules)

            # take the action
            next_state, reward, done = env.step(action)
            reward = np.sign(reward)

            if render:
                env.render()

            temp_buffer.append([state.copy(), int(action), float(reward), next_state.copy(), float(done)])

            if use_rules:
                ep_reward_rule+= reward
            else:
                ep_reward_net+= reward

            num_frames+=1

            # multiply state by 7 through flipping / rotating frame
            for flip in [False,True]:
                for rot in range(1,4):
                    if flip:
                        f_state = np.flip(state.reshape(22,22,3), 1).reshape(3,22,22)
                        f_next_state = np.flip(next_state.reshape(22,22,3),1).reshape(3,22,22)
                    else:
                        f_state = state.copy()
                        f_next_state = next_state.copy()
                    rotated_state = np.rot90(f_state.reshape(22,22,3), rot).reshape(3,22,22) # state gets rotated
                    rotated_next_state = np.rot90(f_next_state.reshape(22,22,3), rot).reshape(3,22,22) # state gets rotated
                    if flip:
                        f_action = act_flip[action]
                    else:
                        f_action = int(action)
                    rotated_action = rotate_action_list[rot-1][f_action] # action gets rotated
                    temp_buffer.append([rotated_state.copy(), int(rotated_action), float(reward), rotated_next_state.copy(), float(done)])

                    num_frames+=1

            if done:
                break

            # update state for next iteration
            state = next_state.copy()

        # discount the rewards before adding to replay buffer
        gamma = 0.99
        R = 0
        for i in reversed(range(len(temp_buffer))):
            _, _, reward, _, done = temp_buffer[i]
            if done:
                R = 0
            R = reward + gamma*R
            temp_buffer[i][2] = float(R)

        # push values to replay buffer
        for info in temp_buffer:
            replay_buffer.push(*info)

        # perform training
        if len(replay_buffer) >= initial_buffer_size and i_episode % 10 == 0:
            print("Training")
            for i in range(10):
                loss_val = update_model(replay_buffer.sample(batch_size))
            num_frames = 0


        # update cumulative reward
        running_reward_rule = 0.05 * ep_reward_rule + (1 - 0.05) * running_reward_rule
        running_reward_net = 0.05 * ep_reward_net + (1 - 0.05) * running_reward_net

        # record average reward and loss
        writer.add_scalar('Training loss', loss_val, i_episode)
        writer.add_scalar('Average reward - rule', running_reward_rule, i_episode)
        writer.add_scalar('Average reward - net', running_reward_net, i_episode)

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage reward - rule: {:.2f}\tAverage reward - net: {:.2f}'.format(
                  i_episode, running_reward_rule, running_reward_net))

        # save model every so often
        if i_episode % 100 == 0:
            print("Saving model")
            torch.save(model.state_dict(), save_net_path)

        # check if we have "solved" the problem
        if running_reward_net > 100:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward_net, t))
            break

if __name__ == '__main__':
    try:
        main()
    finally:
        # save final model
        print("Saving model")
        torch.save(model.state_dict(), save_net_path)
