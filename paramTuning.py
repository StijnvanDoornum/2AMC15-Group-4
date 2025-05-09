from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm
from datetime import datetime
from time import time as now

from world.environment import Environment
from agents import MCAgent

# Configuration
GRID_PATH         = Path("grid_configs/A1_grid.npy")
SIGMA             = 0.40
N_EPISODES        = 5000
MAX_STEPS         = 1000
MOVING_AVG_WINDOW = 300
REWARD_VARIANTS = {
    "default": (-1, -5, 10),
    "mild_penalty": (-0.5, -2, 10),
    "severe_penalty": (-2, -10, 10),
}

# Hyperparameter grid
GAMMAS           = [0.90, 0.95, 0.99]
EPSILONS         = [1.0]
EPSILON_MINS     = [0.05]
EPSILON_DECAYS   = [0.999, 0.9999]

def set_custom_rewards(env: Environment, rewards: tuple[float, float, float]):
    """Monkey patch reward scheme into environment."""
    env.custom_rewards = rewards

    def custom_step(self, action: int):
        old_step = Environment.step.__get__(self)
        s, _, done, info = old_step(action)
        tile_type = self.grid[s[1], s[0]]
        match tile_type:
            case 0: reward = self.custom_rewards[0]
            case 1 | 2: reward = self.custom_rewards[1]
            case 3: reward = self.custom_rewards[2]
        return s, reward, done, info

    env.step = custom_step.__get__(env)


def run_experiment(config_name, gamma, epsilon, epsilon_min, epsilon_decay, rewards):
    env = Environment(GRID_PATH, no_gui=True, sigma=SIGMA, random_seed=2025)
    env.reset() 
    set_custom_rewards(env, rewards)

    agent = MCAgent(
        grid_shape=env.grid.shape,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=2025
    )

    episode_returns = []
    for ep in range(1, N_EPISODES + 1):
        state = env.reset()
        done = False
        G = 0
        step = 0

        while not done and step < MAX_STEPS:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(next_state, reward, info["actual_action"])
            state = next_state
            G += reward
            step += 1

        agent.end_episode()
        episode_returns.append(G)

    return episode_returns


# Run all experiments
results = {}
config_id = 0

all_combinations = list(product(GAMMAS, EPSILONS, EPSILON_MINS, EPSILON_DECAYS, REWARD_VARIANTS.items()))
total_configs = len(all_combinations)

for i, (gamma, epsilon, epsilon_min, epsilon_decay, (reward_name, rewards)) in tqdm(
    enumerate(all_combinations),
    total=total_configs,
    desc="Experiment Progress",
    ncols=100
):
    config_name = (
        f"C{i:03}_"
        f"g{gamma}_e{epsilon}_emin{epsilon_min}_edec{epsilon_decay}_rw_{reward_name}"
    )

    print(f"\n[{i+1}/{total_configs}] Running config: {config_name}")
    t0 = now()
    returns = run_experiment(config_name, gamma, epsilon, epsilon_min, epsilon_decay, rewards)
    elapsed = now() - t0
    print(f"→ Completed in {elapsed:.1f} seconds")

    results[config_name] = returns


# Visualization
plt.figure(figsize=(12, 6))
for config_name, returns in results.items():
    if len(returns) >= MOVING_AVG_WINDOW:
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW, mode="valid")
        plt.plot(ma, label=config_name)
plt.xlabel("Episode")
plt.ylabel("Return (moving avg)")
plt.title("Monte Carlo Agent: Hyperparameter & Reward Comparison")
plt.legend(loc="upper left", fontsize=7)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
Path("results").mkdir(exist_ok=True)
plt.savefig(Path("results") / f"comparison_plot_{timestamp}.png")
plt.show()

print("\nAll experiments completed. Results and plots saved.")
