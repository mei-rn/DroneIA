import gym
from gym import spaces
import numpy as np
import pickle
import os
import pygame
import sys

# colors
GRAY = (150, 150, 150)
WHITE = (200, 200, 200)
DRONE = "drone.png"

# window
WINDOW_SIZE = 800
CELLS = 50  # number of cells in the grid
CELL_SIZE = int(WINDOW_SIZE / CELLS)

# empty map
# MAP = [[0 for _ in range(CELLS)] for _ in range(CELLS)] # uncomment this line to create an empty map

# load map from file (/\ do not comment this function)
def load_map(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
# save map to a file
def save_map(filename):
    with open(filename, 'wb') as f:
        pickle.dump(MAP, f)
        
class DroneGridEnv(gym.Env):
    def __init__(self, grid):
        super(DroneGridEnv, self).__init__()
        self.grid = grid
        self.grid_size = np.array(grid).shape
        self.observation_space = spaces.Tuple((spaces.Discrete(self.grid_size[0]), spaces.Discrete(self.grid_size[1])))
        self.action_space = spaces.Discrete(4)  # 4 discrete actions: 0 = up, 1 = down, 2 = left, 3 = right
        self.start_pos = (self.grid_size[0] - 1, 0)  # Starting position at bottom left corner
        self.goal_pos = (0, self.grid_size[1] - 1)  # Goal position at top right corner
        self.current_pos = self.start_pos  # Initialize current position

    def reset(self):
        self.current_pos = self.start_pos  # Reset current position to start position
        return self.current_pos  # Return initial state

    def step(self, action):
        assert self.action_space.contains(action)
        
        # Define movement based on action
        if action == 0:  # Up
            new_pos = (self.current_pos[0] - 1, self.current_pos[1])
        elif action == 1:  # Down
            new_pos = (self.current_pos[0] + 1, self.current_pos[1])
        elif action == 2:  # Left
            new_pos = (self.current_pos[0], self.current_pos[1] - 1)
        elif action == 3:  # Right
            new_pos = (self.current_pos[0], self.current_pos[1] + 1)
        
        # Check if new position is within bounds and not an obstacle
        if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1] and self.grid[new_pos[0]][new_pos[1]] != 1:
            self.current_pos = new_pos  # Update current position
            
            # Check if goal state is reached
            done = (self.current_pos == self.goal_pos)
            
            # Calculate reward
            if done:
                reward = 1.0  # Positive reward for reaching the goal
                self.reset()
            else:
                reward = 0.0  # No reward for non-goal states
        else:
            done = False  # Episode continues
            reward = -0.1  # Negative reward for colliding with obstacle or going out of bounds
        
        return self.current_pos, reward, done, {}  # Return next state, reward, episode termination flag, and additional info

def main():
    global MAP, WINDOW, clock
    pygame.init()
    WINDOW_SIZE = 800
    WINDOW = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # create a square window
    clock = pygame.time.Clock()

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
                        MAP = load_map(selected_map)  # load selected map
        
                        env = DroneGridEnv(MAP)
                        runEnvironment(env)
        pygame.display.update()
        
def drawMapSelectionScreen(window):
    window.fill(GRAY)  # fill window with gray
    font = pygame.font.SysFont(None, 40)
    text = font.render("Select a Map", True, WHITE)
    text_rect = text.get_rect(center=(WINDOW_SIZE // 2, 50))
    window.blit(text, text_rect)

def drawMapButtons(window):
    map_buttons = []
    map_files = [f for f in os.listdir() if f.startswith("map_") and f.endswith(".pkl")]
    button_names = [f[len("map_"):].rstrip('.pkl') for f in map_files]
    button_height = 50
    for i, (map_file, button_name) in enumerate(zip(map_files, button_names)):
        button_rect = pygame.Rect(20, 100 + i * (button_height + 10), 300, button_height)
        pygame.draw.rect(window, WHITE, button_rect)  # draw button
        pygame.draw.rect(window, GRAY, button_rect, 3)  # draw border
        font = pygame.font.SysFont(None, 30)
        text = font.render(button_name, True, GRAY)
        text_rect = text.get_rect(center=button_rect.center)
        window.blit(text, text_rect)
        map_buttons.append((button_rect, map_file))
    return map_buttons

# Draw the drone
def draw_drone(x, y):
    drone_img = pygame.image.load("drone.png")  # Load the drone image
    drone_size = CELL_SIZE   # Adjust the size of the drone
    drone_img = pygame.transform.scale(drone_img, (drone_size, drone_size))  # Scale the image to match the drone size
    drone_rect = drone_img.get_rect()
    drone_rect.topleft = (x * CELL_SIZE + (CELL_SIZE - drone_size) // 2, y * CELL_SIZE + (CELL_SIZE - drone_size) // 2)
    WINDOW.blit(drone_img, drone_rect)
    

def getClickedButton(buttons, pos):
    for button in buttons:
        if button[0].collidepoint(pos):
            return button
    return None

def runEnvironment(env):
    while True:
        # Handle events
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.KEYDOWN:

                # Check if the pressed key is the 'A' key
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 1
                if event.key == pygame.K_LEFT:
                    action = 2
                if event.key == pygame.K_RIGHT:
                    action = 3
                
        if action != None:
            env.step(action)
        
        # Draw grid
        WINDOW.fill(GRAY)  # fill window with gray
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            for y in range(0, WINDOW_SIZE, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(WINDOW, WHITE, rect, 1)  # draw grid with white borders
                if MAP[y // CELL_SIZE][x // CELL_SIZE] == 1:  # if it's an obstacle cell (1)
                    pygame.draw.rect(WINDOW, WHITE, rect)  # draw obstacle

        # Render the drone position
        draw_drone(env.current_pos[1], env.current_pos[0])

        pygame.display.update()
        clock.tick(60)  # Cap the frame rate at 60 FPS

if __name__ == "__main__":
    main() 