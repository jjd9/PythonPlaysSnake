import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from snakeEnv import GridWorld

env = GridWorld()
env.seed(42)
torch.manual_seed(42)


SavedAction = namedtuple('SavedAction', ['log_prob', 'probs', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)

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

optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), probs, state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    loss_val = 0

    if len(model.rewards) > 30:
        # calculate the true value using rewards returned from the environment
        for r, d in zip(model.rewards[::-1], model.dones[::-1]):
            # calculate the discounted value
            if d:
                R = r
            else:
                R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, probs, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]
        del model.dones[:]
        loss_val = loss.item()

    return loss_val


def main():
    running_reward = 0
    render = False
    log_interval = 10

    plt.ion()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    losses=[]
    avg_rewards=[]
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

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)

            if render:
                env.render()

            model.rewards.append(reward)
            model.dones.append(done)

            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        loss_val = finish_episode()

        avg_rewards.append(running_reward)
        losses.append(loss_val)

        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

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


        if i_episode % 500 == 0 and running_reward > 5:
            print("Saving model")
            torch.save(model.state_dict(), "SnakeNet_AC")

        # check if we have "solved" the cart pole problem
        if running_reward > 100:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

print("Saving model")
torch.save(model.state_dict(), "SnakeNet_AC")

if __name__ == '__main__':
    main()
