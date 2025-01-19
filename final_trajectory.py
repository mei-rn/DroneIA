import numpy as np
import matplotlib.pyplot as plt
import pygame
import os
import sys
import pickle
from main import DroneGridEnv


# define colors
GRAY = (150, 150, 150)
WHITE = (200, 200, 200)
PURPLE = (150, 0, 150)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DRONE = "drone.png"

# window
WINDOW_SIZE = 800


def load(filename):
    ''' Load a map from a file'''
    with open(filename, 'rb') as f:
        return pickle.load(f)

def map_to_env(grid):
    ''' Convert a map to a grid environment'''
    env = DroneGridEnv(grid)
    return env

def find_optimal_path(env, Q_table = None): # Trouver le chemin optimal

    ''' Génère une image du chemin optimal avec la Q_Table donnée'''
    
    if Q_table is None:
        raise ValueError("Q_table is not defined")
    
    if Q_table.shape[0] != env.grid_size[0]*env.grid_size[1]:
        raise ValueError("Q_table shape does not match the grid size")
    
    
    grid = np.array(env.grid)
    optimal_path = np.zeros_like(grid)

    state = env.reset()

    optimal_path[state[1], state[0]] = 1
    iteration = 0

    while state != env.goal_pos:
        state_index = state[0] + state[1] * env.grid_size[0]
        action = np.argmax(Q_table[state_index])
        state, _, done = env.step(action)
        optimal_path[state[1], state[0]] = 1
        
        iteration += 1
        if (done):
            break
        if (iteration > 500):
            break
    
    print(optimal_path)
    return optimal_path

def preprocess_map(grid, agent_pos, goal_pos, path):

    ''' Preprocess the map into a grid image'''

    grid = np.array(grid)

    processed_grid = np.zeros_like(grid)
    processed_grid[grid == 1] = 1  # Obstacle
    processed_grid[goal_pos] = 3  # Goal
    processed_grid[agent_pos] = 2  # Agent

    if path.all() != None:
       
       processed_grid[path == 1] = 4  # Path

       for x in range(grid.shape[0]):
           for y in range(grid.shape[1]):
               if grid[x][y] == 1 and path[x][y] == 1:
                   processed_grid[x][y] = 5  # Obstacle in path

    
    return processed_grid

def render(grid_img, drone_position = None):
    '''Render the grid image with the drone position and the path'''
    
    pygame.init()
    WINDOW = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # create a square window
    # Draw grid
    WINDOW.fill(GRAY)  # fill window with gray

    CELLS = grid_img.shape[0]
    CELL_SIZE = int(WINDOW_SIZE / CELLS)
    while True:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:  # handle window closing (ends program)
                pygame.quit()
                sys.exit()

        for x in range(0, WINDOW_SIZE, CELL_SIZE):

            for y in range(0, WINDOW_SIZE, CELL_SIZE):

                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(WINDOW, WHITE, rect, 1)  # draw grid with white borders

                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 1:  # if it's an obstacle cell (1)
                    pygame.draw.rect(WINDOW, WHITE, rect)  # draw obstacle

           # if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 2: #if drone (2)
                # draw_drone(drone_position[0], drone_position[1])

                
            
                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 4: #if path (4)
                    pygame.draw.rect(WINDOW, GREEN, rect) #draw path in green

                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 3: #if goal (3)
                    pygame.draw.rect(WINDOW, PURPLE, rect) #draw goal in purple
                
                if grid_img[y // CELL_SIZE][x // CELL_SIZE] == 5:
                    pygame.draw.rect(WINDOW, RED, rect)
        
        pygame.display.update()

def choose_image(env,q):
    optimal_path = find_optimal_path(env, q)
    grid_img = preprocess_map(env.grid, env.current_pos, env.goal_pos, optimal_path)
    render(grid_img, env.current_pos)
    
# Load maps, environments and Q-Tables
map_simple = load('map_simple.pkl')
map_mid = load('map_mid.pkl')
map_hard = load('map_hard.pkl')


env_simple = map_to_env(map_simple)
env_mid = map_to_env(map_mid)
env_hard = map_to_env(map_hard)


q_simple = load('Simple - Q-Learning.pkl')
q_mid = load('Mid - Q-Learning.pkl')
q_hard = load('Hard - Q-Learning.pkl')

s_simple = load('Simple - SARSA.pkl')
s_mid = load('Mid - SARSA.pkl')
s_hard = load('Hard - SARSA.pkl')

a_simple = load('Simple - Alternating.pkl')
#a_mid = load('Mid - Alternating.pkl')
#a_hard = load('Hard - Alternating.pkl')

#choose_image(env_simple, q_simple)
#choose_image(env_mid, q_mid)
#choose_image(env_hard, q_hard)

#choose_image(env_simple, s_simple)
#choose_image(env_mid, s_mid)
#choose_image(env_hard, s_hard)

choose_image(env_simple, a_simple)
#choose_image(env_mid, a_mid)
#choose_image(env_hard, a_hard)
