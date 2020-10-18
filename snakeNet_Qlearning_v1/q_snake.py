from Qlearning import Qlearning
from snakeEnv import GridWorld
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

import IPython, sys

state_components = [[-1,0,1], [-1,0,1], [0,1,2], [0,1,2], [0,1,2], [0,1,2]]
possible_states = []
for a in state_components[0]:
    for b in state_components[1]:
        for c in state_components[2]:
            for d in state_components[3]:
                for e in state_components[4]:
                    for f in state_components[5]:
                        if np.sum(np.array([c,d,e,f])==1) <= 1:
                            possible_states.append(str([a,b,c,d,e,f]))
possible_actions = [0,1,2,3]
learning_rate = 0.05
discount_factor = 0.99

training = True

if training:
    q_man = Qlearning(possible_states,
                    possible_actions,
                    learning_rate,
                    discount_factor)
else:
    q_man = pickle.load(open("Q_snake",'rb'))

if not training:
    render = True
else:
    render = False

running_reward = 0
avg_rewards = []
log_interval = 10
env = GridWorld()
T = 1
# run inifinitely many episodes
for i_episode in range(5000):

    # reset environment and episode reward
    state = env.reset()
    ep_reward = 0

    if not training:
        board = env.render(show=False)
        fps = 10
        size = (board.shape[0],board.shape[1])
        out = cv2.VideoWriter("Q_snake.avi",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # for each episode, only run 9999 steps so that we don't 
    # infinite loop while learning
    for t in range(1, 5000):            

        # select action from policy
        if training:
            action = q_man.get_best_action(str(state.astype(int).tolist()), T)
            T+=1
        else:
            out.write(board)
            action = q_man.get_best_action(str(state.astype(int).tolist()), t=0)

        # take the action
        next_state, reward, done = env.step(action)

        if render:
            board = env.render(show=False)

        if training:  
            q_man.update_model(str(state.astype(int).tolist()), action, reward, str(next_state.astype(int).tolist()), done)


        ep_reward += reward
        if done:
            break

        state = next_state.copy()

    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    avg_rewards.append(running_reward)

    if render:
        cv2.destroyAllWindows()
        out.release()
        if ep_reward > 50:
            break


    # log results
    if i_episode % log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tUnvisited Table Entries: {:d}'.format(
                i_episode, ep_reward, running_reward, q_man.unvisited_table_entries()))
    
    if training and (i_episode % 500 == 0 and running_reward > 5):
        pickle.dump(q_man, open("Q_snake",'wb'))

    # check if we have "solved" the cart pole problem
    if running_reward > 100:
        print("Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, t))
        break

if training:
    pickle.dump(q_man, open("Q_snake",'wb'))
