import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
from interaction_module import compute_action_torch
from env_utils.loading import save_weights

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
def optimize(memory, policy_net, target_net, gamma, optimizer, batch_size = 100):

    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1)
    reward_batch = torch.FloatTensor(np.array(reward_batch))
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.FloatTensor(np.array(done_batch))

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def deep_q_learning(env, memory, policy_net, target_net, optimizer, alpha=0.1, gamma=0.9,  epsilon=0.1, epsilon_decay=0.006, target_update_freq = 1000, episodes = 1000):

    steps = 0
    rewards = []
    start_time = time.time()
    #environments_name = ['Simple', 'Mid', 'Hard']

    for e in range(episodes): #Run 1k training runs

        #env = random.choice(environments)
        #print('Environment :', environments_name[environments.index(env)])
        state = env.reset() #Part of OpenAI where you need to reset at the start of each run
        total_reward = 0 #Set initial reward to 0
        step_per_episode = 0
        time_per_episode = time.time()

        while True: #Loop until done == True
            #IF random number is less than epsilon grab the random action else grab the argument max of Q[state]

            #current_state_index = env.current_pos[0] + env.current_pos[1]*env.observation_space[0] # Obtain the index of the state

            current_state = np.copy(env.grid)

            action = compute_action_torch(env, epsilon, policy_net) # Compute the action for the current state in function of the epsilon_greedy

            posp1, new_state, reward, done = env.step(action)

            #state_tp1_index = posp1[0] + posp1[1]*env.observation_space[0]

            memory.append((current_state, action, reward, new_state, done))

            total_reward += reward #Increment your reward

            optimize(memory, policy_net, target_net, gamma, optimizer)

            if steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if step_per_episode >= 4000:
                done = True

            if done:
                print(f"Episode: {e}, Reward: {total_reward}, Steps in the episode: {step_per_episode}", f"Time: {time.time() - time_per_episode}")
                break

            steps += 1
            step_per_episode += 1


        if epsilon>0.1:
            epsilon *= np.exp(-epsilon_decay)

        rewards.append(total_reward)

    delta_time = start_time - time.time()
    print(f"Time: {delta_time}")

    save_weights('policy_net_weights_simple.pth', policy_net)
    save_weights('target_net_weights_simple.pth', target_net)

    return rewards, delta_time, steps