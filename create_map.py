import os
import pygame
import sys
import pickle


# colors
GRAY = (150, 150, 150)
WHITE = (200, 200, 200)
DRONE = "drone.png"

# window
WINDOW_SIZE = 800
CELLS = 50  # number of cells in the grid
CELL_SIZE = int(WINDOW_SIZE / CELLS)

# empty map
MAP = [[0 for _ in range(CELLS)] for _ in range(CELLS)] # uncomment this line to create an empty map

# load map from file (/\ do not comment this function)
def load_map(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
# save map to a file
def save_map(filename):
    with open(filename, 'wb') as f:
        pickle.dump(MAP, f)

# main function
def main():
    global WINDOW, MAP
    pygame.init()
    WINDOW = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # create a square window

    while True:  # runs until interrupted
        drawMapSelectionScreen(WINDOW)
        map_buttons = drawMapButtons(WINDOW)  # draw map selection buttons
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # handle window closing (ends program)
                # save_map('map_gazelle.pkl') # uncomment this line to save map in a file
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:  # handle mouse clicking
                if pygame.mouse.get_pressed()[0]:  # left mouse button
                    pos = pygame.mouse.get_pos()  # get click position
                    clicked_button = getClickedButton(map_buttons, pos)
                    if clicked_button:  # if a button is clicked
                        selected_map = clicked_button[1]
                        MAP = load_map(selected_map)  # load selected map
                        drawMap()  # switch to game screen
        pygame.display.update()

# draw map selection screen
def drawMapSelectionScreen(window):
    window.fill(GRAY)  # fill window with gray
    font = pygame.font.SysFont(None, 40)
    text = font.render("Select a Map", True, WHITE)
    text_rect = text.get_rect(center=(WINDOW_SIZE // 2, 50))
    window.blit(text, text_rect)


# draw map selection buttons
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

# check if a button is clicked
def getClickedButton(buttons, pos):
    for button in buttons:
        if button[0].collidepoint(pos):
            return button
    return None

# draw map
def drawGrid():
    WINDOW.fill(GRAY)  # fill window with gray
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(WINDOW, WHITE, rect, 1)  # draw grid with white borders
            if MAP[y // CELL_SIZE][x // CELL_SIZE] == 1:  # if it's an obstacle cell (1)
                pygame.draw.rect(WINDOW, WHITE, rect)  # draw obstacle

# draw on selected map
def drawMap():
    while True:
        drawGrid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # handle window closing (ends program)  
                #save_map('map_gazelle.pkl') # uncomment this line to save map in a file (et changez le nom svp allez pas override les maps des autres gens >:( )
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



if __name__ == "__main__":
    main()