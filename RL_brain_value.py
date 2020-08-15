
import numpy as np
import pandas as pd
import sys 

class rlalgorithm:
    def __init__(self, actions, env, reward_decay = 0.9):
        self.env = env
        self.actions_count = actions  
        self.gamma = reward_decay
        self.display_name="VI"
        self.epsilon = 0.1
        self.values = np.zeros((10,10)).tolist()
        self.actions = np.full((10,10,4), ['u', 'd', 'l', 'r']).tolist()

    def choose_action(self, s_):
    # implement this. 
    # Choose the next action given the observation/state
    # observation is a unique key to represent a state

        if (s_ == "terminal"):
            return np.random.choice(self.actions_count)
        
        x, y = self.compute_indexes(s_)

        state_action = self.actions[y][x]

        # choose a state action
        action = np.random.choice(state_action)

        # up: 0  
        # down: 1 
        # right: 2 
        # left: 3
        if (action == 'u'):
            return 0
        elif (action == 'd'):
            return 1
        elif (action == 'r'):
            return 2
        elif (action == 'l'):
            return 3

        #action = np.random.choice(self.actions_count)


    def learn(self, s, a, r, s_):
        
        if s_ == 'terminal':
            new_values = np.zeros((10,10))
            
            #Policy evaluation 
            #while True:
                #delta = 0.0
                # for all 100 states 
                
            for i in range(10):
                for j in range(10):
                    if (self.isWall(i, j) or self.isGoal(i, j) or self.isPit(i,j)):
                        continue
                    
                    v = self.values[j][i]
                    new_values[j][i] = self.calculate_value(j, i)
                    #delta = max(delta, abs(v - new_values[j][i]))
                            
            self.values = new_values

                #if delta < self.epsilon:
                    #break
            
            #Policy improvement 
            for i in range(10):
                for j in range(10):
                    if (self.isWall(i, j) or self.isGoal(i, j) or self.isPit(i, j)):
                        continue
                    
                    action_value_pair = dict()
                    for action in ['u', 'd', 'l', 'r']:
                        reward, reverse = self.calculate_reward(i, j, action)

                        if reverse:
                            continue

                        value_next_state = self.calculate_value_next_state(i, j, action) 
                        pi = reward + self.gamma * value_next_state
                        action_value_pair[action] = pi
                    
                    max_value = max(action_value_pair.values())
                    max_policy = []
                    for key, value in action_value_pair.iteritems():
                        if (value == max_value):
                            max_policy.append(key)

                    new_actions = max_policy
                    self.actions[j][i] = new_actions
           
            
        #print("values: ")
        #print(self.values)
        #print("actions: ")
        #print(self.actions)
        
        a_ = self.choose_action(s_)
        return s_, a_
        
        
    def isWall(self, x, y):
        state = self.compute_coordinates(x,y)
        return state in [self.env.canvas.coords(w) for w in self.env.wallblocks]

    def isPit(self, x, y):
        state = self.compute_coordinates(x,y)
        return state in [self.env.canvas.coords(w) for w in self.env.pitblocks]

    def isGoal(self, x, y):
        state = self.compute_coordinates(x,y)
        return state == self.env.canvas.coords(self.env.goal)

    def calculate_value(self, i, j):
        current_coordinates = self.compute_coordinates(i, j)
        reward, _, _ = self.env.computeReward(current_coordinates, 0, current_coordinates)

        action_value_pair = dict()
        possible_actions = self.actions[i][j]

        #probability = self.calculate_probability(len(possible_actions))
        
        for action in possible_actions:
            reward, _ = self.calculate_reward(j, i, action)

            value_next_state = self.calculate_value_next_state(j, i, action) 
            value = reward + self.gamma * value_next_state
            action_value_pair[action] = value
        
        max_value = max(action_value_pair.values())
        
        return round(max_value, 5)

    def calculate_probability(self, action_space_count):
        return 1.0/action_space_count

    def calculate_reward(self, i, j, action):
        current_coordinates = self.compute_coordinates(i, j)
        reward, _, reverse = self.env.computeReward(current_coordinates, 0, current_coordinates)

        if (action == 'u'):
            j_ = j - 1
            if (j_ >= 0):
                next_coordinates = self.compute_coordinates(i, j_)
                reward, _, reverse = self.env.computeReward(current_coordinates, 0, next_coordinates)
        
        elif (action == 'd'):
            j_ = j + 1
            if (j_ <= 9):
                next_coordinates = self.compute_coordinates(i, j_)
                reward, _, reverse = self.env.computeReward(current_coordinates, 0, next_coordinates)

        elif (action == 'l'):
            i_ = i - 1
            if (i_ >= 0):
                next_coordinates = self.compute_coordinates(i_, j)
                reward, _, reverse = self.env.computeReward(current_coordinates, 0, next_coordinates)
        
        elif (action == 'r'):
            i_ = i + 1
            if (i_ <= 9):
                next_coordinates = self.compute_coordinates(i_, j)
                reward, _, reverse = self.env.computeReward(current_coordinates, 0, next_coordinates)

        return reward, reverse
    
    def calculate_value_next_state(self, i, j, action): 
        if (action == 'u'):
            j_ = j - 1
            if (j_ >= 0):
                return self.values[j_][i]

        elif (action == 'd'):
            j_ = j + 1
            if (j_ <= 9):
                return self.values[j_][i]

        elif (action == 'l'):
            i_ = i - 1
            if (i_ >= 0):
                return self.values[j][i_]
            
        elif (action == 'r'):
            i_ = i + 1
            if (i_ <= 9):
                return self.values[j][i_]

        return self.values[j][i]

    def compute_coordinates(self, x, y): 
        UNIT = 40 
        origin = np.array([UNIT/2, UNIT/2])
        center = origin + np.array([UNIT * x, UNIT*y])

        return [center[0] - 15.0, center[1] - 15.0,
            center[0] + 15.0, center[1] + 15.0]
    
    def compute_indexes(self, s_):
        UNIT = 40
        x = (s_[0] + 15 - UNIT/2) / UNIT
        y = (s_[1] + 15 - UNIT/2) / UNIT

        return int(x), int(y) 



