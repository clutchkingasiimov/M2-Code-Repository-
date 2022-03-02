import os, sys
from re import T
import gym
import time
 
import text_flappy_bird_gym

class QLearning:

    def __init__(self,environment,alpha,gamma,epsilon):

        self.actions = environment.action_space.n
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.Q_table = {} #Key: Coordinate position (State), Value: Action


    def initialize_q_values(self,state):
        #Check if the state exists in the table 
        #If not, then load in the initialized value as 0
        if state in self.Q_table is None:
            self.Q_table[state] = [0,0]

    # def agent_start(self,state):

    #     #Choose a random action using epsilon-greedy strategy 
    #     random_action_prob = np.random.random()
    #     if random_action_prob < self.epsilon:
            #Choose a random action to explore 

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()

    # iterate
    while True:

        # Select next action
        action = env.action_space.sample()  # for an agent, action = agent.policy(observation)

        # Appy action and return new observation of the environment
        obs, reward, done, info = env.step(action)
        print(reward)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close() 
