import numpy as np
import time
from agents.utils.interaction import compute_action_numpy

def q_learning(env, alpha=1, gamma=0.99,  epsilon=0.99, epsilon_decay=0.00025, episodes = 10001, max_iter_episode = 500):
    start_time = time.time()
    Q = np.zeros((env.grid_size[0]*env.grid_size[1], len(env.action_space)), dtype=np.float32) #Initialize the Q table to all 0s
    rewards = []
    mean_reward_for_1k_episode = 0

    for e in range(episodes): #Run 1k training runs

        state, _ = env.reset() #Part of OpenAI where you need to reset at the start of each run
        total_reward = 0 #Set initial reward to 0
        iteration = 0

        if e % 1000 == 0:
            mean_reward_for_1k_episode = float(mean_reward_for_1k_episode / 1000)
            rewards.append(mean_reward_for_1k_episode)
            print(f"Episode: {e}, Mean reward: {mean_reward_for_1k_episode}, Epsilon: {epsilon}")
            mean_reward_for_1k_episode = 0


        while True: #Loop until done == True
            #IF random number is less than epsilon grab the random action else grab the argument max of Q[state]

            current_state_index = env.current_pos[0] + env.current_pos[1]*env.observation_space[0] # Obtain the index of the state

            action = compute_action_numpy(current_state_index, Q, epsilon, env) # Compute the action for the current state in function of the epsilon_greedy

            posp1, _, reward, done = env.step(action) #Send your action to OpenAI and get back the tuple

            state_tp1_index = posp1[0] + posp1[1]*env.observation_space[0]

            total_reward += reward #Increment your reward
            mean_reward_for_1k_episode += reward

            Q[current_state_index][action] = Q[current_state_index][action] + alpha * (reward + gamma * np.max(Q[state_tp1_index]) - Q[current_state_index][action])

             #Make sure to keep random at 10%

            if done:
                #print(f"Episode: {e}, Reward: {total_reward}, Epsilon: {epsilon}")
                break

            iteration += 1

            if iteration >= max_iter_episode:
                #print(f"Episode: {e}, Reward: {total_reward}")
                break


        if epsilon>0.1:
            epsilon *= np.exp(-epsilon_decay)

        rewards.append(total_reward)

    delta_time = time.time() - start_time
    print(f"Time: {delta_time}")

    return Q, rewards, delta_time