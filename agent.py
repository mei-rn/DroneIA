import numpy as np
import pickle
import random

class QAgent:
    def __init__(self, state_size, action_size, exploration_proba, time):
        
        self.n_actions = action_size
        #we define some parameters and hyperparameters:
        #"lr" : learning rate
        #"gamma": discounted factor
        #"exploration_proba_decay": decay of the exploration probability
        
        self.lr = 1
        self.gamma = 0.9
        self.exploration_proba = exploration_proba # 1 if epsilon greedy algorithm
        self.exploration_proba_decay = 0.005
        self.state_size = state_size
        self.time = time

        
    # The agent computes the action to perform given a state 
    def compute_action(self, current_pos, Q_table):
        '''
        We sample a variable uniformly over [0,1]
        if the variable is less than the exploration probability
             we choose an action randomly;
        else
             we choose the action with the highest Q-value.
        '''
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        else:
            current_state = current_pos[0] + 25*current_pos[1] #Obtain the value of the state (from 0 to 2499)
            return np.argmax(Q_table[current_state]) #choose the best action according to the Q_table in this state

    # when an episode is finished, we update the exploration probability using espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def train(self, state, Q_table):
        rewards = []
        count = 0
        distance = 0
        for time in range(1000-self.time): # For 1000 epochs
            done = False
            while not done: # While the epoch is not ended
                random_num = random.random() # Random number between 0 and 1
                at = self.compute_action(state.current_pos, Q_table) # Compute the action for the current state
                st = state.current_pos[0] + state.current_pos[1]*self.state_size[0] # Compute the indices of the state
                posp1, reward, done = state.step(at) # Perform the action to obtain the next position, the reward and the status of the environment
                
                if done == True: # If goal is reached
                    Q_table[st][at] = (1 - self.lr) * Q_table[st][at] + self.lr * reward #Update Q_table
                    rewards.append(np.copy(Q_table)) # Save reward into list 
                    print(np.copy(Q_table))
                    count+=1
                    print("Number =", count)
                    break
                
                # Update Q_function
                atp1 = self.compute_action(posp1, Q_table) # Compute the action for the next state
                stp1 = posp1[0] + posp1[1]*self.state_size[0] # Compute the indices of the state at t+1
                state.current_pos = posp1 # Update the position of the agent at t+1 in the environment

                if (random_num <= 0.5): # We use Q-learning
                    Q_table[st][at] = (1-self.lr)*(Q_table[st][at]) 
                    + self.lr * (reward + self.gamma*max(Q_table[stp1])) # Update Q_table with Q-learning
                    distance += 1
                else: # We use SARSA
                    Q_table[st][at] = (1-self.lr)*(Q_table[st][at]) 
                    + self.lr * (reward + self.gamma*Q_table[stp1][atp1]) # Update Q_table with SARSA
                    distance += 1

            self.save_checkpoint('Training1.pkl', Q_table, time) # Save training checkpoint
            self.update_exploration_probability() 
        print("Total distance: ", distance)
        with open('rewards_simple.pkl', 'wb') as f: # Rewards into pkl
            pickle.dump(rewards, f)
        
    def save_checkpoint(self, filename, Q_table, time):
        decayed_explo_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay * time)
        checkpoint = np.array([Q_table, decayed_explo_proba, time], dtype= object)
    
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



# ag = QAgent(state_size=50*50, action_size=4)
# ag.save_table('Q_gael.pkl')
# check = load_checkpoint('Training1.pkl')