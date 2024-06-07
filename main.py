import gym
from gym import spaces
import numpy as np
import pickle
import os
import pygame
import sys
from create_map import drawMapButtons, drawMapSelectionScreen, load_map, getClickedButton
from agent import QAgent, load_checkpoint

# colors
GRAY = (150, 150, 150)
WHITE = (200, 200, 200)
PURPLE = (160, 32, 240)
DRONE = "drone.png"

# window
WINDOW_SIZE = 800
CELLS = 25  # number of cells in the grid
CELL_SIZE = int(WINDOW_SIZE / CELLS)

STATE_SIZE = (25,25)
ACTION_SIZE = 4


# Uncomment to continue training of existing agent
# checkpoint = load_checkpoint('Training1.pkl')
# agent_q = QAgent(STATE_SIZE, ACTION_SIZE, exploration_proba=checkpoint[1], time = checkpoint[2]) #continue training of existing agent
# Q_table = checkpoint[0]

# Uncomment to initialize training of a new agent
Q_table = np.zeros((STATE_SIZE[0]*STATE_SIZE[1], ACTION_SIZE), dtype=np.float32) #Init empty Q_table
agent_q = QAgent(STATE_SIZE, ACTION_SIZE, exploration_proba=1, time = 0)

class DroneGridEnv(gym.Env):
    def __init__(self, grid):
        super(DroneGridEnv, self).__init__()
        self.grid = grid
        self.grid_size = np.array(grid).shape
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_size[0]), spaces.Discrete(self.grid_size[1])))
        self.action_space = spaces.Discrete(4)  # 4 discrete actions: 0 = up, 1 = down, 2 = left, 3 = right
        self.start_pos = (0, 0)  # Starting position at top left corner
        self.goal_pos = (self.grid_size[0] - 1, self.grid_size[1] - 1)  # Goal position at bottom right corner
        self.current_pos = self.start_pos  # Initialize current position

    def reset(self):
        self.current_pos = self.start_pos  # Reset current position to start position
        return self.current_pos  # Return initial state

    def step(self, action):
        assert self.action_space.contains(action)
        
        
        # Define movement based on action
        if action == 0:  # Up
            new_pos = (self.current_pos[0], self.current_pos[1] - 1)
        elif action == 1:  # Down
            new_pos = (self.current_pos[0], self.current_pos[1] + 1)
        elif action == 2:  # Left
            new_pos = (self.current_pos[0] - 1, self.current_pos[1])
        elif action == 3:  # Right
            new_pos = (self.current_pos[0] + 1, self.current_pos[1])
        
        # Check if new position is within bounds and not an obstacle
        if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1] and self.grid[new_pos[1]][new_pos[0]] != 1:
            self.current_pos = new_pos  # Update current position
            
            # Check if goal state is reached
            done = (self.current_pos == self.goal_pos)
            
            # Calculate reward
            if done:
                reward = 100.0  # Positive reward for reaching the goal
                self.reset()
                
            elif self.grid[new_pos[1]][new_pos[0]] == 1: 
                reward = -100 # Negative reward for going in a wall
            
            else:
                reward = 0 #Negative reward for non-goal state
            
        else:
            done = False
            reward = -100  # Negative reward for going out of bounds
        
        grid_image = preprocess_map(self.grid, self.current_pos, self.goal_pos) #process the map into a grid image
        drone_pos = self.current_pos #collect the drone position
        render(grid_image, drone_pos) # Rendering the environment
        
        return self.current_pos, reward, done  # Return next state, reward, episode termination flag


def main():
    global MAP, WINDOW, Q_table
    pygame.init()
    WINDOW_SIZE = 800
    WINDOW = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # create a square window

    while True:  # runs until interrupted
        drawMapSelectionScreen(WINDOW)
        map_buttons = drawMapButtons(WINDOW)  # draw map selection buttons
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # handle window closing (ends program)
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:  # handle mouse clicking
                if pygame.mouse.get_pressed()[0]:  # left mouse button
                    pos = pygame.mouse.get_pos()  # get click position
                    clicked_button = getClickedButton(map_buttons, pos)
                    if clicked_button:  # if a button is clicked
                        selected_map = clicked_button[1]
                        print(selected_map)
                        MAP = load_map(selected_map)  # load selected map
                        env = DroneGridEnv(MAP)
                        print(env)
                        runEnvironment(env, Q_table)
                        
        pygame.display.update()



# Draw the drone
def draw_drone(x, y):
    drone_img = pygame.image.load("drone.png")  # Load the drone image
    drone_size = CELL_SIZE   # Adjust the size of the drone
    drone_img = pygame.transform.scale(drone_img, (drone_size, drone_size))  # Scale the image to match the drone size
    drone_rect = drone_img.get_rect() # Get the surface to display the drone
    drone_rect.topleft = (x * CELL_SIZE + (CELL_SIZE - drone_size) // 2, y * CELL_SIZE + (CELL_SIZE - drone_size) // 2)
    WINDOW.blit(drone_img, drone_rect) # Display the drone
    

def runEnvironment(env, Q_table):
    while True:
        # Handle events
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.KEYDOWN:

                # Check if the pressed key
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 1
                if event.key == pygame.K_LEFT:
                    action = 2
                if event.key == pygame.K_RIGHT:
                    action = 3
                
                if event.key == pygame.K_t:
                    agent_q.train(env, Q_table) # Launch the training of the agent
                
        if action != None:
            env.step(action) # If an action is taken by the player
        
        grid_image = preprocess_map(env.grid, env.current_pos, env.goal_pos) #process the map into a grid image
        render(grid_image, env.current_pos) # Rendering the environment

def render(grid_img, drone_position):
    # Draw grid
        WINDOW.fill(GRAY)  # fill window with gray
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(WINDOW, WHITE, rect, 1)  # draw grid with white borders

                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 1:  # if it's an obstacle cell (1)
                    pygame.draw.rect(WINDOW, WHITE, rect)  # draw obstacle

                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 2: #if drone (2)
                    draw_drone(drone_position[0], drone_position[1]) #draw drone

                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 3: #if goal (3)
                    pygame.draw.rect(WINDOW, PURPLE, rect) #draw goal

        
        pygame.display.update()

def preprocess_map(grid, agent_pos, goal_pos):
    grid = np.array(grid)
    processed_grid = np.zeros_like(grid)
    processed_grid[grid == 1] = 1  # Obstacle
    processed_grid[goal_pos] = 3  # Goal
    processed_grid[agent_pos] = 2  # Agent
    return processed_grid

if __name__ == "__main__":
    main() 