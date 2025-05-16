import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from train_q_learning import train_agent as train_q_learning
from train_mc import train_agent as train_mc
from train_VI import main as train_vi
from agents import ValueIterationAgent
from world.environment import Environment
from agents import QLearningAgent
from agents import MCAgent
import argparse
import itertools

# Configuration
MOVING_AVG_WINDOW = 50

def parse_args():
    parser = argparse.ArgumentParser(description="Run different types of experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "reward_comparison",
            "agent_comparison",
            "hyperparameter_comparison",
            "agent_comparison_with_optimal_rewards",
            "epsilon_comparison",
            "optimal_epsilon_comparison",
            "reward_grid_comparison",
            "q_mc_selected_epsilon"
        ],
        default="agent_comparison",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["q_learning", "monte_carlo", "value_iteration"],
        default="q_learning",
        help="Agent type to use (only used for reward_comparison, hyperparameter_comparison, and epsilon_comparison)"
    )
    return parser.parse_args()

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
    plt.legend()
    plt.tight_layout()
    plt.show()


def custom_reward_function(empty_reward: float, wall_reward: float, target_reward: float):
    def reward_fn(grid, agent_pos):
        cell_value = grid[agent_pos]
        if cell_value == 0:  # Empty tile
            return empty_reward
        elif cell_value in (1, 2):  # Wall/obstacle
            return wall_reward
        elif cell_value == 3:  # Target
            return target_reward
        return empty_reward  # Default case
    return reward_fn


def compare_reward_parameters(
    agent_type: str = "q_learning",
    grid_path: Path = Path("grid_configs/A1_grid.npy"),
    episodes: int = 3000,
    max_steps: int = 600,
    sigma: float = 0.1,
    random_seed: int = 2025
):
    # Define the reward parameter grid
    empty_rewards = [-0.1, -1]
    wall_rewards = [-1, -5, -10, -15]
    target_rewards = [5, 10, 20]  
    results = {}
    for empty_reward, wall_reward, target_reward in itertools.product(
        empty_rewards, wall_rewards, target_rewards
    ):
        tag = f"empty{empty_reward}_wall{wall_reward}_target{target_reward}"
        print(f"\nRunning {agent_type} with reward parameters: {tag}")
        reward_fn = custom_reward_function(
            empty_reward=empty_reward,
            wall_reward=wall_reward,
            target_reward=target_reward
        )
        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=sigma,
            target_fps=20,
            reward_fn=reward_fn,
            random_seed=random_seed
        )
        start_time = time.time()
        if agent_type == "q_learning":
            state = env.reset()
            agent = QLearningAgent(
                grid_shape=env.grid.shape,
                alpha=0.20,
                gamma=0.90,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                seed=random_seed
            )
            returns = []
            for ep in range(episodes):
                state = env.reset()
                done = False
                G = 0
                step = 0
                while not done and step < max_steps:
                    step += 1
                    action = agent.take_action(state)
                    next_s, reward, done, info = env.step(action)
                    agent.update(next_s, reward, info["actual_action"])
                    state = next_s
                    G += reward
                agent.end_episode()
                returns.append(G)
                if ep % 100 == 0:
                    print(f"Episode {ep}: Return = {G:.2f}")
        elif agent_type == "monte_carlo":
            from agents import MCAgent
            state = env.reset()
            agent = MCAgent(
                grid_shape=env.grid.shape,
                gamma=1.0,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.999,
                seed=random_seed
            )
            returns = []
            for ep in range(episodes):
                state = env.reset()
                done = False
                G = 0
                step = 0
                while not done and step < max_steps:
                    step += 1
                    action = agent.take_action(state)
                    next_s, reward, done, info = env.step(action)
                    agent.update(next_s, reward, info["actual_action"])
                    state = next_s
                    G += reward
                agent.end_episode()
                returns.append(G)
                if ep % 100 == 0:
                    print(f"Episode {ep}: Return = {G:.2f}")
        elif agent_type == "value_iteration":
            from agents import ValueIterationAgent
            agent = ValueIterationAgent(gamma=0.90)
            agent._start_state = env.reset()
            agent.train(env)
            # Evaluate the policy for a number of episodes
            eval_episodes = 1000
            returns = []
            for _ in range(eval_episodes):
                state = env.reset()
                done = False
                G = 0
                step = 0
                while not done and step < max_steps:
                    step += 1
                    action = agent.take_action(state)
                    next_s, reward, done, info = env.step(action)
                    state = next_s
                    G += reward
                returns.append(G)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        results[tag] = {
            "returns": returns,
            "final_return": np.mean(returns[-100:]),
            "agent": agent,
            "params": {
                "empty_reward": empty_reward,
                "wall_reward": wall_reward,
                "target_reward": target_reward
            }
        }
    # Print final results
    print("\nFinal Results (average of last 100 episodes):")
    for name, result in results.items():
        print(f"{name}: {result['final_return']:.2f}")
    # Plot all learning curves
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    color_idx = 0
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_return'], reverse=True)
    for tag, result in sorted_results:
        returns = result["returns"]
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
        plt.plot(ma, 
                 color=colors[color_idx],
                 lw=1.5, 
                 label=f"{tag} (final: {result['final_return']:.1f})")
        color_idx += 1
    plt.title(f"{agent_type.replace('_', ' ').title()} Performance with Different Reward Parameters\n(episodes={episodes}, max_steps={max_steps}, σ={sigma})", pad=20)
    plt.xlabel("Episode")
    plt.ylabel("Return (50-ep Moving Average)")
    plt.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / f"reward_parameter_comparison_{agent_type}.pdf", 
                dpi=300, 
                bbox_inches='tight')
    plt.show()


def compare_hyperparameters(
    agent_type: str,
    grid_path: Path = Path("grid_configs/A1_grid.npy"),
    episodes: int = 3000,
    max_steps: int = 600,
    sigma: float = 0.1,
    random_seed: int = 2025
):
    if agent_type == "q_learning":
        
        alphas = [0.05, 0.10, 0.15, 0.20]
        gammas = [0.60, 0.80, 0.90, 0.99]
        eps0 = 1.0
        eps_decay = 0.995
        results = {}
        for alpha, gamma in itertools.product(alphas, gammas):
            tag = f"a{alpha}_g{gamma}_e{eps0}_d{eps_decay}_s{sigma}_gridA1_grid"
            print(f"\nRunning {agent_type} with {tag}")
            start_time = time.time()
            agent, returns = train_q_learning(
                grid_path=grid_path,
                episodes=episodes,
                max_steps=max_steps,
                sigma=sigma,
                random_seed=random_seed,
                alpha=alpha,
                gamma=gamma,
                epsilon=eps0,
                epsilon_min=0.05,
                epsilon_decay=eps_decay
            )
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
            results[tag] = {
                "returns": returns,
                "final_return": np.mean(returns[-100:]),
                "agent": agent
            }
        # Print final results and plot (as before)
        print("\nFinal Results (average of last 100 episodes):")
        for name, result in results.items():
            print(f"{name}: {result['final_return']:.2f}")
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(alphas) * len(gammas)))
        color_idx = 0
        for alpha, gamma in itertools.product(alphas, gammas):
            tag = f"a{alpha}_g{gamma}_e{eps0}_d{eps_decay}_s{sigma}_gridA1_grid"
            if tag not in results:
                continue
            returns = results[tag]["returns"]
            ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid")
            plt.plot(ma, 
                     color=colors[color_idx],
                     lw=1.5, 
                     label=f"α={alpha}, γ={gamma}")
            color_idx += 1
        plt.title(f"Q-learning Performance on A1 Grid\n(ε₀={eps0}, decay={eps_decay}, σ={sigma})", pad=20)
        plt.xlabel("Episode")
        plt.ylabel("Return (50-ep Moving Average)")
        plt.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        Path("figs").mkdir(exist_ok=True)
        plt.savefig(Path("figs") / "all_curves_A1_grid.pdf", 
                    dpi=300, 
                    bbox_inches='tight')
        plt.show()
    elif agent_type == "monte_carlo":
        from train_mc import train_agent as train_mc_agent
        epsilon_decays = [0.999, 0.9995, 0.9999]
        max_steps_list = [250, 350, 500]
        episodes_mc = 10000
        results = {}
        for epsilon_decay, max_steps in itertools.product(epsilon_decays, max_steps_list):
            tag = f"eps_decay={epsilon_decay}, max_steps={max_steps}"
            print(f"\nRunning MC with {tag}")
            agent, returns = train_mc_agent(
                grid_path=grid_path,
                sigma=0.0,
                episodes=episodes_mc,
                max_steps=max_steps,
                gamma=1.0,
                epsilon=1.0,
                epsilon_min=0.00,
                epsilon_decay=epsilon_decay,
                random_seed=random_seed,
                show_gui=False,
                save_results=False
            )
            results[tag] = returns
        # Plot
        plt.figure(figsize=(8, 4))
        for tag, returns in results.items():
            plt.plot(range(len(returns)), returns, label=tag)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Convergence of the Monte Carlo Agent between different settings of max_steps and epsilon_decay")
        plt.legend(fontsize=7)
        plt.tight_layout()
        Path("figs").mkdir(exist_ok=True)
        plt.savefig(Path("figs") / "mc_hyperparameter_convergence.pdf", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Only Q-learning and Monte Carlo are supported for hyperparameter tuning")


def compare_agents():
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
        sigma=0,
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

def compare_agents_with_optimal_rewards(
    q_reward_sets, mc_reward_sets, vi_reward_sets,
    grid_path=Path("grid_configs/A1_grid.npy"),
    episodes=3000,
    max_steps=600,
    sigma=0.1,
    random_seed=2025
):
    agent_configs = {
        "Q-learning": q_reward_sets,
        "Monte Carlo": mc_reward_sets,
        "Value Iteration": vi_reward_sets
    }
    results = {}
    for agent_name, reward_sets in agent_configs.items():
        reward_curves = []
        for params in reward_sets:
            reward_fn = custom_reward_function(
                empty_reward=params["empty_reward"],
                wall_reward=params["wall_reward"],
                target_reward=params["target_reward"]
            )
            env = Environment(
                grid_fp=grid_path,
                no_gui=True,
                sigma=sigma,
                target_fps=20,
                reward_fn=reward_fn,
                random_seed=random_seed
            )
            state = env.reset()
            if agent_name == "Q-learning":
                agent = QLearningAgent(
                    grid_shape=env.grid.shape,
                    alpha=0.20,
                    gamma=0.90,
                    epsilon=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.995,
                    seed=random_seed
                )
                returns = []
                for ep in range(episodes):
                    state = env.reset()
                    done = False
                    G = 0
                    step = 0
                    while not done and step < max_steps:
                        step += 1
                        action = agent.take_action(state)
                        next_s, reward, done, info = env.step(action)
                        agent.update(next_s, reward, info["actual_action"])
                        state = next_s
                        G += reward
                    agent.end_episode()
                    returns.append(G)
            elif agent_name == "Monte Carlo":
                from agents import MCAgent
                agent = MCAgent(
                    grid_shape=env.grid.shape,
                    gamma=1.0,
                    epsilon=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.999,
                    seed=random_seed
                )
                returns = []
                for ep in range(episodes):
                    state = env.reset()
                    done = False
                    G = 0
                    step = 0
                    while not done and step < max_steps:
                        step += 1
                        action = agent.take_action(state)
                        next_s, reward, done, info = env.step(action)
                        agent.update(next_s, reward, info["actual_action"])
                        state = next_s
                        G += reward
                    agent.end_episode()
                    returns.append(G)
            elif agent_name == "Value Iteration":
                from agents import ValueIterationAgent
                agent = ValueIterationAgent(gamma=0.90)
                agent._start_state = env.reset()
                agent.train(env)
                eval_episodes = 1000
                returns = []
                for _ in range(eval_episodes):
                    state = env.reset()
                    done = False
                    G = 0
                    step = 0
                    while not done and step < max_steps:
                        step += 1
                        action = agent.take_action(state)
                        next_s, reward, done, info = env.step(action)
                        state = next_s
                        G += reward
                    returns.append(G)
            reward_curves.append((params, returns))
        results[agent_name] = reward_curves
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (agent_name, reward_curves) in enumerate(results.items()):
        ax = axes[idx]
        for params, returns in reward_curves:
            ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
            label = f"empty={params['empty_reward']}, wall={params['wall_reward']}, target={params['target_reward']}"
            ax.plot(ma, lw=1.5, label=label)
        ax.set_title(agent_name)
        ax.set_xlabel("Episode")
        if idx == 0:
            ax.set_ylabel("Return (50-ep MA)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle("Agent Comparison: Multiple Reward Functions")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / "agent_comparison_multiple_rewards.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def compare_agents_on_multiple_rewards(
    reward_sets,
    grid_path=Path("grid_configs/A1_grid.npy"),
    episodes=3000,
    max_steps=600,
    sigma=0.1,
    random_seed=2025
):
    agent_names = ["Q-learning", "Monte Carlo", "Value Iteration"]
    results_per_reward = []
    for params in reward_sets:
        reward_fn = custom_reward_function(
            empty_reward=params["empty_reward"],
            wall_reward=params["wall_reward"],
            target_reward=params["target_reward"]
        )
        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=sigma,
            target_fps=20,
            reward_fn=reward_fn,
            random_seed=random_seed
        )
        state = env.reset()
        agent_results = {}
        # Q-learning
        agent = QLearningAgent(
            grid_shape=env.grid.shape,
            alpha=0.20,
            gamma=0.90,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            seed=random_seed
        )
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                agent.update(next_s, reward, info["actual_action"])
                state = next_s
                G += reward
            agent.end_episode()
            returns.append(G)
        agent_results["Q-learning"] = returns
        # Monte Carlo
        from agents import MCAgent
        agent = MCAgent(
            grid_shape=env.grid.shape,
            gamma=1.0,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.999,
            seed=random_seed
        )
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                agent.update(next_s, reward, info["actual_action"])
                state = next_s
                G += reward
            agent.end_episode()
            returns.append(G)
        agent_results["Monte Carlo"] = returns
        # Value Iteration
        from agents import ValueIterationAgent
        agent = ValueIterationAgent(gamma=0.90)
        agent._start_state = env.reset()
        agent.train(env)
        eval_episodes = 1000
        returns = []
        for _ in range(eval_episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                state = next_s
                G += reward
            returns.append(G)
        agent_results["Value Iteration"] = returns
        results_per_reward.append(agent_results)
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (params, agent_results) in enumerate(zip(reward_sets, results_per_reward)):
        ax = axes[idx]
        for agent_name in agent_names:
            returns = agent_results[agent_name]
            ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
            ax.plot(ma, lw=1.5, label=agent_name)
        ax.set_title(f"empty={params['empty_reward']}, wall={params['wall_reward']}, target={params['target_reward']}")
        ax.set_xlabel("Episode")
        if idx == 0:
            ax.set_ylabel("Return (50-ep MA)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.suptitle("Agent Comparison on Different Reward Functions")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / "agent_comparison_multiple_rewards.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def compare_epsilon_parameters(
    agent_type: str,
    grid_path: Path = Path("grid_configs/A1_grid.npy"),
    episodes: int = 3000,
    max_steps: int = 600,
    sigma: float = 0.1,
    random_seed: int = 2025
):
    # Define epsilon parameter grid
    epsilons = [1.0, 0.5]
    epsilon_mins = [0.05, 0.01]
    epsilon_decays = [0.995, 0.999]
    results = {}
    for epsilon, epsilon_min, epsilon_decay in itertools.product(epsilons, epsilon_mins, epsilon_decays):
        tag = f"eps={epsilon}_min={epsilon_min}_decay={epsilon_decay}"
        print(f"\nRunning {agent_type} with {tag}")
        # Use default reward function
        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=sigma,
            target_fps=20,
            random_seed=random_seed
        )
        state = env.reset()
        if agent_type == "q_learning":
            agent = QLearningAgent(
                grid_shape=env.grid.shape,
                alpha=0.20,
                gamma=0.90,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                seed=random_seed
            )
        elif agent_type == "monte_carlo":
            from agents import MCAgent
            agent = MCAgent(
                grid_shape=env.grid.shape,
                gamma=1.0,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                seed=random_seed
            )
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                agent.update(next_s, reward, info["actual_action"])
                state = next_s
                G += reward
            agent.end_episode()
            returns.append(G)
        results[tag] = returns
    # Plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for idx, (tag, returns) in enumerate(results.items()):
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
        plt.plot(ma, color=colors[idx], lw=1.5, label=tag)
    plt.title(f"{agent_type.replace('_', ' ').title()} Performance for Different Epsilon Settings (Default Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Return (50-ep MA)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / f"epsilon_comparison_{agent_type}_default_reward.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def compare_agents_with_optimal_epsilon(
    q_epsilon_params, mc_epsilon_params,
    grid_path=Path("grid_configs/A1_grid.npy"),
    episodes=3000,
    max_steps=600,
    sigma=0.1,
    random_seed=2025
):
    agent_names = ["Q-learning", "Monte Carlo"]
    results = {}
    # Q-learning
    reward_fn = custom_reward_function(empty_reward=-0.1, wall_reward=-1, target_reward=10)
    env = Environment(
        grid_fp=grid_path,
        no_gui=True,
        sigma=sigma,
        target_fps=20,
        reward_fn=reward_fn,
        random_seed=random_seed
    )
    state = env.reset()
    agent = QLearningAgent(
        grid_shape=env.grid.shape,
        alpha=0.20,
        gamma=0.90,
        epsilon=q_epsilon_params["epsilon"],
        epsilon_min=q_epsilon_params["epsilon_min"],
        epsilon_decay=q_epsilon_params["epsilon_decay"],
        seed=random_seed
    )
    returns = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        G = 0
        step = 0
        while not done and step < max_steps:
            step += 1
            action = agent.take_action(state)
            next_s, reward, done, info = env.step(action)
            agent.update(next_s, reward, info["actual_action"])
            state = next_s
            G += reward
        agent.end_episode()
        returns.append(G)
    results["Q-learning"] = returns
    # Monte Carlo
    from agents import MCAgent
    agent = MCAgent(
        grid_shape=env.grid.shape,
        gamma=1.0,
        epsilon=mc_epsilon_params["epsilon"],
        epsilon_min=mc_epsilon_params["epsilon_min"],
        epsilon_decay=mc_epsilon_params["epsilon_decay"],
        seed=random_seed
    )
    returns = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        G = 0
        step = 0
        while not done and step < max_steps:
            step += 1
            action = agent.take_action(state)
            next_s, reward, done, info = env.step(action)
            agent.update(next_s, reward, info["actual_action"])
            state = next_s
            G += reward
        agent.end_episode()
        returns.append(G)
    results["Monte Carlo"] = returns
    # Plot
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        returns = results[agent_name]
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
        plt.plot(ma, lw=1.5, label=agent_name)
    plt.title("Q-learning vs. Monte Carlo: Optimal Epsilon Settings (Default Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Return (50-ep MA)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / "agent_comparison_optimal_epsilon_default_reward.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def compare_q_mc_selected_epsilon(
    grid_path=Path("grid_configs/A1_grid.npy"),
    episodes=3000,
    max_steps=600,
    sigma=0.1,
    random_seed=2025
):
    # Q-learning settings
    q_epsilon_settings = [
        {"epsilon": 0.5, "epsilon_min": 0.01, "epsilon_decay": 0.999},
        {"epsilon": 0.5, "epsilon_min": 0.01, "epsilon_decay": 0.995},
        {"epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995},
    ]
    # Monte Carlo settings
    mc_epsilon_settings = [
        {"epsilon": 0.5, "epsilon_min": 0.01, "epsilon_decay": 0.999},
        {"epsilon": 0.5, "epsilon_min": 0.01, "epsilon_decay": 0.995},
        {"epsilon": 0.5, "epsilon_min": 0.05, "epsilon_decay": 0.995},
    ]
    # Q-learning experiment
    q_curves = []
    for params in q_epsilon_settings:
        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=sigma,
            target_fps=20,
            random_seed=random_seed
        )
        state = env.reset()
        agent = QLearningAgent(
            grid_shape=env.grid.shape,
            alpha=0.20,
            gamma=0.90,
            epsilon=params["epsilon"],
            epsilon_min=params["epsilon_min"],
            epsilon_decay=params["epsilon_decay"],
            seed=random_seed
        )
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                agent.update(next_s, reward, info["actual_action"])
                state = next_s
                G += reward
            agent.end_episode()
            returns.append(G)
        q_curves.append((params, returns))
    # Monte Carlo experiment
    mc_curves = []
    from agents import MCAgent
    for params in mc_epsilon_settings:
        env = Environment(
            grid_fp=grid_path,
            no_gui=True,
            sigma=sigma,
            target_fps=20,
            random_seed=random_seed
        )
        state = env.reset()
        agent = MCAgent(
            grid_shape=env.grid.shape,
            gamma=1.0,
            epsilon=params["epsilon"],
            epsilon_min=params["epsilon_min"],
            epsilon_decay=params["epsilon_decay"],
            seed=random_seed
        )
        returns = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            G = 0
            step = 0
            while not done and step < max_steps:
                step += 1
                action = agent.take_action(state)
                next_s, reward, done, info = env.step(action)
                agent.update(next_s, reward, info["actual_action"])
                state = next_s
                G += reward
            agent.end_episode()
            returns.append(G)
        mc_curves.append((params, returns))
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    # Q-learning subplot
    for params, returns in q_curves:
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
        label = f"eps={params['epsilon']}, min={params['epsilon_min']}, decay={params['epsilon_decay']}"
        axes[0].plot(ma, lw=1.5, label=label)
    axes[0].set_title("Q-learning: Epsilon Settings")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return (50-ep MA)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    # Monte Carlo subplot
    for params, returns in mc_curves:
        ma = np.convolve(returns, np.ones(MOVING_AVG_WINDOW)/MOVING_AVG_WINDOW, mode="valid") if len(returns) >= MOVING_AVG_WINDOW else returns
        label = f"eps={params['epsilon']}, min={params['epsilon_min']}, decay={params['epsilon_decay']}"
        axes[1].plot(ma, lw=1.5, label=label)
    axes[1].set_title("Monte Carlo: Epsilon Settings")
    axes[1].set_xlabel("Episode")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.suptitle("Q-learning vs. Monte Carlo: Selected Epsilon Settings")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(Path("figs") / "q_mc_selected_epsilon_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    
    if args.experiment == "reward_comparison":
        compare_reward_parameters(args.agent)
    
    elif args.experiment == "agent_comparison":
        compare_agents()
    
    elif args.experiment == "agent_comparison_with_optimal_rewards":
        compare_agents_with_optimal_rewards(    
        # Optimal reward setting candidates
        q_reward_sets=[
            {"empty_reward": -1, "wall_reward": -5, "target_reward": 20},
            {"empty_reward": -1, "wall_reward": -10, "target_reward": 20},
            {"empty_reward": -1, "wall_reward": -5, "target_reward": 10},
            {"empty_reward": -1, "wall_reward": -10, "target_reward": 10},
        ],
        mc_reward_sets=[
            {"empty_reward": -0.1, "wall_reward": -1, "target_reward": 5},
            {"empty_reward": -1, "wall_reward": -1, "target_reward": 20},
            {"empty_reward": -0.1, "wall_reward": -10, "target_reward": 5},
            {"empty_reward": -1, "wall_reward": -1, "target_reward": 5},
        ],
        vi_reward_sets=[
            {"empty_reward": -0.1, "wall_reward": -1, "target_reward": 5},
            {"empty_reward": -1, "wall_reward": -10, "target_reward": 20},
            {"empty_reward": -1, "wall_reward": -15, "target_reward": 20},
            {"empty_reward": -0.1, "wall_reward": -5, "target_reward": 5},
        ]
)
    
    elif args.experiment == "hyperparameter_comparison":
        if args.agent == "q_learning":
            compare_hyperparameters(args.agent)
        elif args.agent == "monte_carlo":
            compare_hyperparameters(args.agent)
    
    elif args.experiment == "epsilon_comparison":
        compare_epsilon_parameters(args.agent)
    
    elif args.experiment == "optimal_epsilon_comparison":
        compare_agents_with_optimal_epsilon(
            # Optimal epsilon setting candidates
            q_epsilon_params={"epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.995},
            mc_epsilon_params={"epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.999}
        )
    
    elif args.experiment == "q_mc_selected_epsilon":
        compare_q_mc_selected_epsilon()
    