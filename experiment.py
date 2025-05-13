import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from tqdm import trange
from train_q_learning import train_agent as train_q_learning
from train_mc import train_agent as train_mc
from train_VI import main as train_vi
from agents import ValueIterationAgent
from world.environment import Environment

# Configuration
MOVING_AVG_WINDOW = 50

def plot_learning_curves(results: dict[str, dict], episodes: int):
    """Plot learning curves."""
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        returns = result["returns"]
        plt.plot(returns, alpha=0.2)
        
        if len(returns) >= MOVING_AVG_WINDOW:
            ma = np.convolve(
                returns,
                np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
                mode="valid"
            )
            plt.plot(
                range(MOVING_AVG_WINDOW - 1, episodes),
                ma,
                label=name,
                linewidth=2
            )
    
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-learning vs Monte Carlo vs Value Iteration Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_experiments():
    # Common hyperparameters
    episodes = 1000
    max_steps = 600
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    grid_path = Path("grid_configs/A1_grid.npy")
    
    results = {}
    
    # Run Q-learning
    print("\nRunning Q-learning experiment...")
    q_agent, q_returns = train_q_learning(
        grid_path=grid_path,
        episodes=episodes,
        max_steps=max_steps,
        alpha=0.15,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        save_results=True
    )
    
    results["Q-learning"] = {
        "returns": q_returns,
        "final_return": np.mean(q_returns[-100:]),
        "agent": q_agent
    }
    
    # Run Monte Carlo
    print("\nRunning Monte Carlo experiment...")
    mc_agent, mc_returns = train_mc(
        grid_path=grid_path,
        episodes=episodes,
        max_steps=max_steps,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        save_results=True
    )
    
    results["Monte Carlo"] = {
        "returns": mc_returns,
        "final_return": np.mean(mc_returns[-100:]),
        "agent": mc_agent
    }
    
    # Run Value Iteration 
    print("\nRunning Value Iteration experiment...")
    start_time = time.time()
    
    vi_agent, vi_returns = train_vi(
        grid_paths=[grid_path],
        no_gui=True,
        iters=episodes,
        fps=30,
        sigma=0.1,
        random_seed=2025
    )
    
    end_time = time.time()
    print(f"Value Iteration training + evaluation completed in {end_time - start_time:.2f} seconds.")
    
    results["Value Iteration"] = {
        "returns": vi_returns,
        "final_return": np.mean(vi_returns[-100:]),
        "agent": vi_agent
    }
    
    # Print final results
    print("\nFinal Results (average of last 100 episodes):")
    for name, result in results.items():
        print(f"{name}: {result['final_return']:.2f}")
    
    # Plot all learning curves
    plot_learning_curves(results, episodes)


if __name__ == "__main__":
    run_experiments() 