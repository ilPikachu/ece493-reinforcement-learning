import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.03, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.lamda = 0.9
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="sarsa lambda"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        observation = str(observation)
        self.check_state_exist(observation)
 
        #BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
            
            state_action = self.q_table.loc[observation, :]

            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
         
            action = np.random.choice(self.actions)
        return action


    '''Update the Q(S,A) state-action value table using sarsa lamda
    '''
    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)

        self.check_state_exist(s_)

        a_ = self.choose_action(str(s_))

        delta = r + self.gamma * self.q_table.loc[s_, a_] - self.q_table.loc[s, a]
        self.e_table.loc[s, a] += 1
        
        
        for state_index, _ in self.q_table.iterrows():
            #print("index", state_index)
            #print("state", state)
            for action_index in range(0, 4):
                if (self.e_table.loc[state_index, action_index] < 0.5):
                    continue
                self.q_table.loc[state_index, action_index] += self.lr * delta * self.e_table.loc[state_index, action_index]
                self.e_table.loc[state_index, action_index] = self.gamma * self.lamda * self.e_table.loc[state_index, action_index]

        return s_, a_


    '''States are dynamically added to the Q(S,A), E(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        
        if state not in self.e_table.index:
            # append new state to e table
            self.e_table = self.e_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.e_table.columns,
                    name=state,
                )
            )
