import pygame
import sys
import pickle

# colors
GRAY = (150, 150, 150)
WHITE = (200, 200, 200)
DRONE = "drone.png"

# window
WINDOW_SIZE = 800
CELLS = 50 # number of cells in the grid
CELL_SIZE = int(WINDOW_SIZE/CELLS)

# empty map
# MAP = [[0 for _ in range(CELLS)] for _ in range(CELLS)] # uncomment this line to create an empty map

# or load map from file
def load_map(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

MAP = load_map('map_ananas.pkl') # uncomment this line to load a map
    
# main function
def main():
    global WINDOW, MAP
    pygame.init()
    WINDOW = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE)) # create a square window

    while True: # runs until interrupted
        drawGrid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # handle window closing (ends program)  
                save_map('map_ananas.pkl') # uncomment this line to save map in a file (et changez le nom svp allez pas override les maps des autres gens >:( )
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:  # handle mouse clicking
                 if pygame.mouse.get_pressed()[0]:  # left mouse button
                    pos = pygame.mouse.get_pos() # get click position
                    cell_x, cell_y = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE
                    if MAP[cell_y][cell_x] == 0:  # if the cell is not an obstacle
                        MAP[cell_y][cell_x] = 1  # mark it as an obstacle
                    else:
                        MAP[cell_y][cell_x] = 0  # remove the obstacle
        pygame.display.update()

# draw map
def drawGrid():
    WINDOW.fill(GRAY)  # fill window with gray
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(WINDOW, WHITE, rect, 1) # draw grid with white borders
            if MAP[y // CELL_SIZE][x // CELL_SIZE] == 1:  # if it's an obstacle cell (1)
                pygame.draw.rect(WINDOW, WHITE, rect)  # draw obstacle

# save the map to a file
def save_map(filename):
    with open(filename, 'wb') as f:
        pickle.dump(MAP, f)

if __name__ == "__main__":
    main()