import numpy as np

def compute_action_numpy(current_state, Q_table, epsilon, environment):

    if np.random.uniform(0,1) < epsilon:
        return np.random.choice(range(len(environment.action_space)))

    else:
        return np.argmax(Q_table[current_state])
    
def compute_action_torch(environment, epsilon, policy_net):

    if np.random.uniform(0,1) < epsilon:
        return np.random.choice(range(len(environment.action_space))) # Exploration

    else:
        current_state = torch.FloatTensor(environment.grid).unsqueeze(0)
        q = policy_net(current_state)
        return torch.argmax(q).item() # Exploit
    