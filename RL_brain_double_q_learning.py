import numpy as np
import pandas as pd
import random


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table_alpha = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_beta = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="double q learning"

    '''Choose the next action to take given the observed state using an epsilon greedy policy of the sum of both q tables'''
    def choose_action(self, observation):
        observation = str(observation)
        self.check_state_exist(observation)
 
        #BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
            
            state_action_alpha = self.q_table_alpha.loc[observation, :]
            state_action_beta = self.q_table_beta.loc[observation, :]
            #print("q1", state_action_alpha)
            #print("q2", state_action_beta)
            
            state_action_sum = []
            for action_index, action_value in enumerate(state_action_alpha):
                state_action_sum.append(action_value + state_action_beta[action_index])

            #print("action sum", state_action_sum)
            same_max_value_action_index = []
            for index, max in enumerate((state_action_sum == np.max(state_action_sum))):
                if max:
                    same_max_value_action_index.append(index)

            action = np.random.choice(same_max_value_action_index)

        else:
            action = np.random.choice(self.actions)

        return action


    '''Update the Q(S,A) state-action value table using double q learning
    '''
    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)

        self.check_state_exist(s_)

        a_ = self.choose_action(str(s_))
        #print ("action", a_)

        if (self.condition_fifty_percent_probability()):
            max_action_index = self.compute_max_q(s_, self.q_table_alpha)
            self.q_table_alpha.loc[s, a] += self.lr * (r + self.gamma * self.q_table_beta.loc[s_, max_action_index] - self.q_table_alpha.loc[s, a])
        else:
            max_action_index = self.compute_max_q(s_, self.q_table_beta)
            self.q_table_beta.loc[s, a] += self.lr * (r + self.gamma * self.q_table_alpha.loc[s_, max_action_index] - self.q_table_beta.loc[s, a])
            
        return s_, a_

    def condition_fifty_percent_probability(self):
        return (True if random.choice([0, 1]) == 1 else False)

    def compute_max_q(self, s_, q_table):
       state_action = q_table.loc[s_, :]
       #print ("state action", state_action)
       action = np.random.choice(state_action[state_action == np.max(state_action)].index)
       #print ("max action index", action)

       return action

    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table_alpha.index:
            # append new state to q table
            self.q_table_alpha = self.q_table_alpha.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table_alpha.columns,
                    name=state,
                )
            )

        if state not in self.q_table_beta.index:
            # append new state to q table
            self.q_table_beta = self.q_table_beta.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table_beta.columns,
                    name=state,
                )
            )
