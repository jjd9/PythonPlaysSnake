import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import cv2
import sys
import IPython
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from snakeEnv import GridWorld

from collections import deque
import random

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/snakeNet_CNN_DQN', flush_secs=60)


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on: ", device)

temp_model = Policy()
temp_model.load_state_dict(torch.load("SnakeNet_CNN_DQN.pt"))
temp_model.eval()
temp_model.to(device)

target_model = Policy()
target_model.load_state_dict(torch.load("SnakeNet_CNN_DQN.pt"))
target_model.eval()
target_model.to(device)

temp_optimizer = optim.Adam(temp_model.parameters(), lr=7e-4)

eps = np.finfo(np.float32).eps.item()

def select_action(state, ucb):
    with torch.no_grad():
        state = torch.from_numpy(state.copy()).float().unsqueeze(0)
        action_vals = temp_model(state.to(device)) + ucb.to(device)
        action = torch.argmax(action_vals)        
        return action.item()

def finish_episode(batch):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """

    gamma = 0.99
    state, action, reward, next_state, done = batch

    state      = torch.FloatTensor(np.float32(state))
    next_state = torch.FloatTensor(np.float32(next_state))
    action     = torch.LongTensor(action).reshape(-1,1)
    reward     = torch.FloatTensor(reward).reshape(-1,1)
    done       = torch.FloatTensor(done).reshape(-1,1)

    values = temp_model(state.to(device)).gather(1,action.to(device))
    next_values = target_model(next_state.to(device)).max(1).values.reshape(-1,1).detach()

    expected_values = reward.to(device) + gamma * next_values * (1 - done.to(device))
    loss = (values - expected_values).pow(2).mean()
    
    # perform backprop
    temp_optimizer.zero_grad()
    loss.backward()
    temp_optimizer.step()

    # reset rewards and action buffer
    loss_val = loss.item()

    return loss_val


def main():
    running_reward = 0
    render = False
    log_interval = 10
    plotting = False

    # Exploration settings
    N_visits = torch.ones((4,))
    c = 10.0 # exploration coefficient
    min_c = 1.0
    c_decay = 0.999
    number_of_frames = 1

    # setup replay buffer
    batch_size = 32
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

    # updating target model every so many episodes
    update_target_freq = 5

    losses=[]
    avg_rewards=[]

    if plotting:
        plt.ion()    
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        line0, = ax1.plot([],[])
        line1, = ax2.plot([],[])
        ax1.set_ylabel("Losses")
        ax2.set_ylabel("Average Rewards")
        ax1.grid()
        ax2.grid()

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        c = max(min_c, c_decay*c)

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):
            if t % 200 == 0 and ep_reward == 0:
              env.food = np.array(env.randomPosition())
              env.update_state()
              state = env.state

            # select action from policy
            c = 0.0
            ucb = c*torch.sqrt(np.log(number_of_frames)/N_visits)
            action = select_action(state, ucb)

            # update ucb parameters
            N_visits[action] += 1
            number_of_frames += 1

            # take the action
            next_state, reward, done = env.step(action)
            reward = np.sign(reward)

            if render:
                env.render()


            replay_buffer.push(state.copy(), int(action), float(reward), next_state.copy(), float(done))

            ep_reward += reward

            # perform training
            if len(replay_buffer) >= initial_buffer_size:
                loss_val = finish_episode(replay_buffer.sample(batch_size))
            else:
                loss_val = 0.0

            if done:
                break

            # multiply state by 7 through flipping / rotating frame
            for flip in [False,True]:
                for rot in range(1,4):
                    if flip:
                        f_state = np.flip(state.reshape(20,20,3), 1).reshape(3,20,20)
                        f_next_state = np.flip(next_state.reshape(20,20,3),1).reshape(3,20,20)
                    else:
                        f_state = state.copy()
                        f_next_state = next_state.copy()
                    rotated_state = np.rot90(f_state.reshape(20,20,3), rot).reshape(3,20,20) # state gets rotated
                    rotated_next_state = np.rot90(f_next_state.reshape(20,20,3), rot).reshape(3,20,20) # state gets rotated
                    if flip:
                        f_action = act_flip[action]
                    else:
                        f_action = int(action)
                    rotated_action = rotate_action_list[rot-1][f_action] # action gets rotated
                    replay_buffer.push(rotated_state.copy(), int(rotated_action), float(reward), rotated_next_state.copy(), float(done))

            # update state for next iteration
            state = next_state.copy()


        # update target model
        if len(replay_buffer) >= initial_buffer_size:
            if i_episode % update_target_freq == 0:
                # copy temp model parameters into target model
                target_model.load_state_dict(temp_model.state_dict())
                target_model.eval()

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # record average reward and loss
        avg_rewards.append(running_reward)
        if loss_val == 0 and len(losses)>0:
            losses.append(losses[-1])
        else:
            losses.append(loss_val)
        writer.add_scalar('Training loss', loss_val, i_episode)
        writer.add_scalar('Average reward', running_reward, i_episode)

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print(N_visits)
            if plotting:
                line0.set_ydata(losses)
                line1.set_ydata(avg_rewards)
                line0.set_xdata(np.arange(len(losses)))
                line1.set_xdata(np.arange(len(avg_rewards)))
                if len(losses) > 0:
                    ax1.set_xlim(0,len(losses))
                    ax2.set_xlim(0,len(losses))
                    ax1.set_ylim(np.min(losses), np.max(losses))
                    ax2.set_ylim(np.min(avg_rewards), np.max(avg_rewards))

                fig.canvas.draw()
                plt.pause(0.1)

        # save model every so often
        if i_episode % 1000 == 0:
            print("Saving model")
            torch.save(target_model.state_dict(), "SnakeNet_CNN_DQN.pt")

        # check if we have "solved" the problem
        if running_reward > 100:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

# save final model
print("Saving model")
torch.save(target_model.state_dict(), "SnakeNet_CNN_DQN.pt")

if __name__ == '__main__':
    try:
        main()
    finally:
        # save final model
        print("Saving model")
        torch.save(target_model.state_dict(), "SnakeNet_CNN_DQN.pt")
