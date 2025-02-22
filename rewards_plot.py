import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
import seaborn as sns

''' This script is used to generate a plot of each episode's reward over time. '''

rewards_files = [f for f in os.listdir() if f.startswith("Rewards") and f.endswith(".pkl")]
rewards_names = [f[len("Rewards - "):].rstrip('.pkl') for f in rewards_files]
saving_folder = r"C:\Users\gagou\Desktop\Data science and info\Projet Drone\Journaux"

print(rewards_files)
print(rewards_names)

for rewards_file, rewards_name in zip(rewards_files, rewards_names):
    with open(rewards_file, 'rb') as f:
        rewards = pkl.load(f)
        plt.figure(figsize=(15,8))
        ax = sns.lineplot(rewards, label=rewards_file, linewidth=0.5)
        ax.set(xlabel = 'Episode', ylabel = 'Reward', ylim=(-10, 2))
        plt.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Optimum Reward')
        ax.set_yticks(np.arange(-10, 3, step=1))
        ax.set_title(f'{rewards_name} Reward Over Time')
        ax.get_legend().set_visible(False)
        plt.savefig(os.path.join(saving_folder, f'{rewards_name}_reward_plot.png'))
        plt.legend()
        plt.show()

        
'''
plt.legend(rewards_names)
plt.show()'''


