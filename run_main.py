from maze_env import Maze
from RL_brainsample_PI import rlalgorithm as rlalg1
from RL_brain_policy import rlalgorithm as rlalg2
from RL_brain_value import rlalgorithm as rlalg3
from RL_brain_sarsa import rlalgorithm as rlalg4
from RL_brain_q_learning import rlalgorithm as rlalg5
from RL_brain_expected_sarsa import rlalgorithm as rlalg6
from RL_brain_double_q_learning import rlalgorithm as rlalg7
from RL_brain_sarsa_lambda import rlalgorithm as rlalg8
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(experiments):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress Task 1", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    plt.show()

def update(env, RL, data, episodes=50):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward

    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        # RL choose action based on state
        action = RL.choose_action(state)
        counter = 0
        while True:
            # fresh env
            if(showRender or (episode % renderEveryNth)==0):
                env.render(sim_speed)

            counter += 1

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))
            
            if (counter > 1000):
                done = True

            if done:
                state_ = "terminal"

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(state, action, reward, state_)
            #print(counter)

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    env.destroy()

if __name__ == "__main__":
    sim_speed = 0

    #Example Short Fast for Debugging
    showRender=True
    episodes=1000
    renderEveryNth=5
    printEveryNth=1
    do_plot_rewards=True

    #Example Full Run, you may need to run longer
    #showRender=False
    #episodes=2000
    #renderEveryNth=10000
    #printEveryNth=100
    #do_plot_rewards=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications

    agentXY=[0,0]
    goalXY=[4,4]

    # Task 1
    wall_shape=np.array([[2,2],[3,6]])
    pits=np.array([[6,3],[1,4]])
    
    # Task 2
    
    wall_shape=np.array([[6,2],[5,2],[4,2],[3,2],[2,2],[6,3],[6,4],[6,5],
        [2,3],[2,4],[2,5]])
    pits=[]

    # Task 3
    
    wall_shape=np.array([[6,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],
        [3,4],[3,5],[3,6],[4,6],[5,6],[5,7],[7,3]])
    pits=np.array([[1,3],[0,5], [7,7], [8,5]])
    
    

    """
    # example ignore
    env1 = Maze(agentXY,goalXY,wall_shape, pits)
    RL1 = rlalg1(actions=list(range(env1.n_actions)))
    data1={}
    env1.after(10, update(env1, RL1, data1, episodes))
    env1.mainloop()
    experiments = [(env1,RL1, data1)]
    """
    
    # policy
    env2 = Maze(agentXY,goalXY,wall_shape,pits)
    RL2 = rlalg2(actions=list(range(env2.n_actions)), env = env2)
    data2={}
    env2.after(10, update(env2, RL2, data2, episodes))
    env2.mainloop()
    experiments=[(env2,RL2, data2)]
    
    # value
    env3 = Maze(agentXY,goalXY,wall_shape,pits)
    RL3 = rlalg3(actions=list(range(env3.n_actions)), env = env3)
    data3={}
    env3.after(10, update(env3, RL3, data3, episodes))
    env3.mainloop()
    experiments.append((env3,RL3, data3))
    
    # sarsa
    env4 = Maze(agentXY,goalXY,wall_shape,pits)
    RL4 = rlalg4(actions=list(range(env4.n_actions)))
    data4={}
    env4.after(10, update(env4, RL4, data4, episodes))
    env4.mainloop()
    experiments= [(env4,RL4, data4)]

    # expected sarsa
    env6 = Maze(agentXY,goalXY,wall_shape,pits)
    RL6 = rlalg6(actions=list(range(env6.n_actions)))
    data6={}
    env6.after(10, update(env6, RL6, data6, episodes))
    env6.mainloop()
    experiments.append((env6,RL6, data6))
    
   # q learning
    env5 = Maze(agentXY,goalXY,wall_shape,pits)
    RL5 = rlalg5(actions=list(range(env5.n_actions)))
    data5={}
    env5.after(10, update(env5, RL5, data5, episodes))
    env5.mainloop()
    experiments.append((env5,RL5, data5))
    
    # double q-learning
    env7 = Maze(agentXY,goalXY,wall_shape,pits)
    RL7 = rlalg7(actions=list(range(env7.n_actions)))
    data7={}
    env7.after(10, update(env7, RL7, data7, episodes))
    env7.mainloop()
    experiments.append((env7,RL7, data7))

    # sarsa lamda
    env8 = Maze(agentXY,goalXY,wall_shape,pits)
    RL8 = rlalg8(actions=list(range(env8.n_actions)))
    data8={}
    env8.after(10, update(env8, RL8, data8, episodes))
    env8.mainloop()
    experiments.append((env8,RL8, data8))
    
    print("All experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))


    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments)

    #Not implemented yet
    #if(do_save_data):
    #    for env, RL, data in experiments:
    #        saveData(env,RL,data)

