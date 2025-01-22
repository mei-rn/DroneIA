import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
import seaborn as sns

''' This script is used to generate a plot of each episode's reward over time. '''

rewards_files = [f for f in os.listdir() if f.startswith("Rewards") and f.endswith(".pkl")]
rewards_names = [f[len("Rewards - "):].rstrip('.pkl') for f in rewards_files]


print(rewards_files)
print(rewards_names)

for rewards_file, rewards_name in zip(rewards_files, rewards_names):
    with open(rewards_file, 'rb') as f:
        rewards = pkl.load(f)
        plt.figure(figsize=(15,8))
        ax = sns.lineplot(rewards, label=rewards_file, linewidth=0.5)
        ax.set(xlabel = 'Episode', ylabel = 'Reward', ylim=(-10, 2))
        ax.set_title(f'{rewards_name} Reward Over Time')
        ax.get_legend().set_visible(False)
        plt.show()

'''
plt.legend(rewards_names)
plt.show()'''


