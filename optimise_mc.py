from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from world.environment import Environment
from agents import MCAgent

# Configuration
GRID_PATH         = Path("grid_configs/A1_grid.npy")
SIGMA             = 0
N_EPISODES        = 40_000
MAX_STEPS         = 500
MOVING_AVG_WINDOW = 30000
VIZ_INTERVAL      = 1000  # Set >N_EPISODES to avoid GUI during sweeps

# Hyperparameter sweeps
EPSILON_DECAYS   = [0.999, 0.9995, 0.9999]
MAX_STEPSS      = [250, 350, 500]

# Store results
results = {}

for eps_decay in EPSILON_DECAYS:
    for max_steps in MAX_STEPSS:
            label = f"eps_decay={eps_decay}, max_steps={max_steps}"

            print(f"\nTraining with {label} ...")

            train_env = Environment(GRID_PATH, no_gui=True, sigma=SIGMA, random_seed=2025)
            state     = train_env.reset()

            agent = MCAgent(
                grid_shape=train_env.grid.shape,
                gamma=1,
                epsilon=1, epsilon_min=0.05,
                epsilon_decay=eps_decay,
                seed=2025
            )

            episode_returns = []

            for ep in tqdm(range(1, N_EPISODES + 1), desc=f"{label}", ncols=100):
                state = train_env.reset()
                done  = False
                G     = 0
                step  = 0

                while not done and step < max_steps:
                    step += 1
                    action = agent.take_action(state)
                    next_s, reward, done, info = train_env.step(action)

                    agent.update(next_s, reward, info["actual_action"])
                    state = next_s
                    G += reward

                agent.end_episode()
                episode_returns.append(G)

            # Compute moving average
            if len(episode_returns) >= MOVING_AVG_WINDOW:
                moving_avg = np.convolve(
                    episode_returns,
                    np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
                    mode="valid"
                )
                results[label] = moving_avg


# Plot all moving averages
plt.figure(figsize=(10, 5))
for label, avg in results.items():
    plt.plot(
        range(MOVING_AVG_WINDOW - 1, N_EPISODES),
        avg,
        label=label
    )

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Q‑learning: Moving Averages for Different Epsilon Settings")
plt.legend()
plt.tight_layout()
plt.show()
