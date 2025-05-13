from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser

from world.environment import Environment
from agents import MCAgent


def parse_args():
    p = ArgumentParser(description="Monte Carlo Reinforcement Learning Trainer.")
    p.add_argument("--grid", type=Path, default=Path("grid_configs/A1_grid.npy"),
                   help="Path to the grid file to use.")
    p.add_argument("--sigma", type=float, default=0.0,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--episodes", type=int, default=100_000,
                   help="Number of episodes to train for.")
    p.add_argument("--max_steps", type=int, default=300,
                   help="Maximum steps per episode.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount factor for Monte Carlo.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial exploration rate.")
    p.add_argument("--epsilon_min", type=float, default=0.0,
                   help="Minimum exploration rate.")
    p.add_argument("--epsilon_decay", type=float, default=0.999,
                   help="Exploration rate decay.")
    p.add_argument("--random_seed", type=int, default=2025,
                   help="Random seed value for the environment.")
    return p.parse_args()

# visualization parameters
VIZ_INTERVAL = 300     # run a GUI episode every … episodes
MOVING_AVG_WINDOW = 1000 # size of moving‑average window for the plot


def run_gui_episode(agent: MCAgent,
                    grid_fp: Path,
                    sigma: float,
                    max_steps: int) -> None:
    """Play one full episode with the GUI so you can watch the policy."""
    env = Environment(
        grid_fp=grid_fp,
        no_gui=False,          # turning the GUI on
        sigma=sigma,
        target_fps=20,         # slow enough to see, fast enough to finish
        random_seed=None,
    )

    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.take_action(state)      # there is no learning during showcase, just the optimal policy
        state, _, done, _ = env.step(action)
        steps += 1

    time.sleep(1)
    env.gui.close()


def train_agent(
    grid_path: Path,
    sigma: float = 0.0,
    episodes: int = 100_000,
    max_steps: int = 300,
    gamma: float = 1.0,
    epsilon: float = 1.0,
    epsilon_min: float = 0.0,
    epsilon_decay: float = 0.999,
    random_seed: int = 2025,
    show_gui: bool = False,
    save_results: bool = True
) -> tuple[MCAgent, list[float]]:
    # setting up the environment
    train_env = Environment(grid_path, no_gui=True, sigma=sigma, random_seed=random_seed)
    state = train_env.reset()

    agent = MCAgent(
        grid_shape=train_env.grid.shape,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=random_seed
    )

    episode_returns: list[float] = []

    # training loop
    for ep in tqdm(range(1, episodes + 1), desc="Training", ncols=100):
        done  = False
        G     = 0
        step  = 0

        while not done and step < max_steps:
            step   += 1
            action  = agent.take_action(state)
            next_s, reward, done, info = train_env.step(action)

            agent.update(next_s, reward, info["actual_action"])
            state = next_s
            G    += reward

        agent.end_episode()
        episode_returns.append(G)

        # print every 100 episodes
        if ep == 1 or ep % 100 == 0:
            ma = (np.mean(episode_returns[-MOVING_AVG_WINDOW:])
                  if len(episode_returns) >= MOVING_AVG_WINDOW else np.nan)
            tqdm.write(
                f"Episode {ep:>4} | Return {G:>6.1f} | "
                f"MovingAvg {ma:>6.1f} | ε {agent.epsilon:5.3f} | MAX_STEPS {max_steps:>4} | "
            )

        # show with GUI
        if show_gui and ep % VIZ_INTERVAL == 0:
            print(f"\n Visualising policy after episode {ep} …")
            run_gui_episode(agent, grid_path, sigma, max_steps)
            print("Training continues …\n")

    if save_results:
        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        Path("results").mkdir(exist_ok=True)
        np.save(Path("results") / f"q_table_{timestamp}.npy", agent.Q)
        print(f"\nTraining finished – Q‑table saved as results/q_table_{timestamp}.npy")

    return agent, episode_returns


def plot_learning_curve(returns: list[float], episodes: int, window_size: int = MOVING_AVG_WINDOW):
    """Plot the learning curve with moving average."""
    plt.figure(figsize=(8, 4))
    plt.plot(returns, alpha=0.3)
    if len(returns) >= window_size:
        ma = np.convolve(
            returns,
            np.ones(window_size) / window_size,
            mode="valid"
        )
        plt.plot(
            range(window_size - 1, episodes),
            ma,
            label=f"{window_size}-episode moving average"
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Monte Carlo learning curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    
    agent, returns = train_agent(
        grid_path=args.grid,
        sigma=args.sigma,
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        random_seed=args.random_seed,
        show_gui=True
    )
    
    plot_learning_curve(returns, args.episodes)
