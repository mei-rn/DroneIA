import pickle
import os
import sys
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from main import DroneGridEnv
from main import main

def train():
    tune.run(
        SACTrainer,  # Use the SAC trainer from RLlib
        name="DroneTraining",  # Name of the training run
        checkpoint_freq=100,  # Frequency of checkpoint saving
        checkpoint_at_end=True,  # Save a checkpoint at the end of training
        local_dir="./ray_results/",  # Directory to save results
        config={
            "env": DroneGridEnv,  # Use the DroneGridEnv environment
            "num_workers": 30,  # Number of parallel workers
            "num_cpus_per_worker": 0.5,  # CPU resources per worker
            "env_config": {
                "grid": grid,  # Pass the map grid
                # Add any additional environment configuration here
                },
            },
        stop={"timesteps_total": 5_000_000},  # Stop training after this many timesteps
        )



if __name__ == "__main__":
    main()
    train()