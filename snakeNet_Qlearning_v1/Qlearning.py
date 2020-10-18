import pandas as pd
import numpy as np
import IPython, sys

class Qlearning:
    _qmatrix = None
    _learn_rate = None
    _discount_factor = None

    def __init__(self,
                 possible_states,
                 possible_actions,
                 learning_rate,
                 discount_factor):
        """
        Initialise the q learning class with an initial matrix and the parameters for learning.

        :param possible_states: list of states the agent can be in
        :param possible_actions: list of actions the agent can perform
        :param initial_reward: the initial Q-values to be used in the matrix
        :param learning_rate: the learning rate used for Q-learning
        :param discount_factor: the discount factor used for Q-learning
        """
        # Initialize the matrix with Q-values
        init_data = [[np.random.uniform(-1,1) for _ in possible_states]
                     for _ in possible_actions]
        self._qmatrix = pd.DataFrame(data=init_data,
                                     index=possible_actions,
                                     columns=possible_states)
        # Initialize the visited table with zero
        visit_data = [[1.0 for _ in possible_states]
                     for _ in possible_actions]
        self._visit_table = pd.DataFrame(data=visit_data,
                                     index=possible_actions,
                                     columns=possible_states)

        # Save the parameters
        self._learn_rate = learning_rate
        self._discount_factor = discount_factor

    def unvisited_table_entries(self):
        return np.sum(np.array(self._visit_table)==1)

    def upper_Confidence(self, state, t):
        """Return upper confidence score for each action at a given state."""
        c = 2.0
        N_visits = self._visit_table[[state]]
        Q_vals = self._qmatrix[[state]]        
        U_vals = Q_vals + c*np.sqrt(np.log(t)/N_visits)
        return U_vals

    def get_best_action(self, state, t):
        """
        Retrieve the action resulting in the highest Q-value for a given state.

        :param state: the state for which to determine the best action
        :return: the best action from the given state
        """        
        if t == 0:
            action = self._qmatrix[[state]].idxmax().iloc[0]
        else:
            U_vals = self.upper_Confidence(state, t)
            action = U_vals.idxmax().iloc[0]
            self._visit_table.at[action, state] += 1

        return action

    def update_model(self, state, action, reward, next_state, done):
        """
        Update the Q-values for a given observation.

        :param state: The state the observation started in
        :param action: The action taken from that state
        :param reward: The reward retrieved from taking action from state
        :param next_state: The resulting next state of taking action from state
        """
        # Update q_value for a state-action pair Q(s,a):
        # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
        q_sa = self._qmatrix.loc[action, state]
        max_q_sa_next = self._qmatrix.loc[self.get_best_action(next_state, 0), next_state]
        r = reward
        alpha = self._learn_rate
        gamma = self._discount_factor

        # Do the computation
        new_q_sa = q_sa + alpha * (r + (gamma * max_q_sa_next * (1-done)) - q_sa)
        self._qmatrix.at[action, state] = new_q_sa
