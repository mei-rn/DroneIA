import numpy as np
final_Q_table

def find_optimal_path(Q_table, env): #Trouver le chemin optimal
    path = []
    state = env.reset()
    path.append(state)

    while state != env.goal_pos:
        state_index = state[0] + state[1] * env.grid_size[0]
        action = np.argmax(Q_table[state_index])
        state, _, done = env.step(action)
        path.append(state)
        if done:
            break

    return path

optimal_path = find_optimal_path(Q_table)