import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the input size, hidden size, and output size
input_size = 6
hidden_size = 16
output_size = 4

# Create an instance of the NeuralAgent class
agent = NeuralAgent(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

def train_neural(agent, env, num_episodes):
    # Train the agent for a specified number of episodes
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()

        # Initialize the episode reward
        episode_reward = 0

        # Loop through the episode
        while True:
            # Convert the state to a PyTorch tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            action_probs = agent(state_tensor)

            # Sample an action from the action probabilities
            action = torch.multinomial(action_probs, num_samples=1).item()

            # Take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Update the episode reward
            episode_reward += reward

            # Convert the next state to a PyTorch tensor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Calculate the target tensor
            with torch.no_grad():
                next_action_probs = agent(next_state_tensor)
                target = reward + 0.99 * torch.max(next_action_probs)

            # Calculate the loss
            loss = criterion(action_probs[0, action], target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the state
            state = next_state

            # Check if the episode is done
            if done:
                break

        # Print the episode reward every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f'Episode [{episode+1}/{num_episodes}], Episode Reward: {episode_reward:.2f}')

def calculate_input_tensor(grid, agent_position):
    # Get the dimensions of the grid
    height, width = grid.shape

    # Get the agent's position
    x, y = agent_position

    # Initialize the input tensor with zeros
    input_tensor = np.zeros(6)

    # Calculate the distance to the nearest obstacle in the left direction
    for i in range(x-1, -1, -1):
        if grid[i, y] == 1:
            input_tensor[0] = x - i
            break

    # Calculate the distance to the nearest obstacle in the right direction
    for i in range(x+1, height):
        if grid[i, y] == 1:
            input_tensor[1] = i - x
            break

    # Calculate the distance to the nearest obstacle in the up direction
    for j in range(y-1, -1, -1):
        if grid[x, j] == 1:
            input_tensor[2] = y - j
            break

    # Calculate the distance to the nearest obstacle in the down direction
    for j in range(y+1, width):
        if grid[x, j] == 1:
            input_tensor[3] = j - y
            break

    # Calculate the distance to the goal in the horizontal direction
    goal_position = np.argwhere(grid == 2)[0]
    input_tensor[4] = goal_position[1] - y

    # Calculate the distance to the goal in the vertical direction
    input_tensor[5] = goal_position[0] - x

    # Convert the input tensor to a PyTorch tensor
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    return input_tensor

def calculate_reward(grid, agent_position, num_moves):
    # Get the dimensions of the grid
    height, width = grid.shape

    # Get the agent's position
    x, y = agent_position

    # Initialize the reward
    reward = 0

    # Check if the agent has collided with an obstacle
    if grid[x, y] == 1:
        reward -= 10

    # Check if the agent has reached the goal
    if grid[x, y] == 2:
        reward += 100

    # Calculate the distance to the goal
    goal_position = np.argwhere(grid == 2)[0]
    distance_to_goal = np.sqrt((goal_position[0] - x)**2 + (goal_position[1] - y)**2)

    # Reward the agent for moving closer to the goal
    reward += 1 / distance_to_goal

    return reward

