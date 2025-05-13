import time
from pathlib import Path
import numpy as np

from world.environment import Environment
from agents import QLearningAgent, MCAgent, ValueIterationAgent


def evaluate_steps(agent, env: Environment, max_steps: int) -> int:
    """
    Run one greedy episode from the given start state and return number of steps to termination.
    """
    state = env.reset()
    done = False
    steps = 0
    rewards = 0
    while not done and steps < max_steps:
        steps += 1
        action = agent.take_action(state)
        state, reward, done, _ = env.step(action)
        rewards += reward
    return steps, rewards


def train_on_grid(grid_fp: Path,
                  sigma: float,
                  seed: int,
                  n_episodes: int,
                  max_steps: int) -> dict:
    """
    Train Q-Learning, Monte Carlo, and Value Iteration agents on a single grid.
    Returns timing and step-count metrics for each agent.
    """
    # Prepare environment and fixed start state
    env = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma, random_seed=seed, agent_start_pos=(10,3))
    env.reset()

    results = []

    # Define agent configurations
    configs = [
        ('ValueIteration', ValueIterationAgent, {}),
        ('Q-Learning', QLearningAgent, {'alpha': 0.15, 'gamma': 0.95,
                                        'epsilon': 1.0, 'epsilon_min': 0.05, 'epsilon_decay': 0.995}),
        ('MonteCarlo', MCAgent, {'gamma': 1.0,
                                 'epsilon': 1.0, 'epsilon_min': 0.05, 'epsilon_decay': 0.999})
    ]

    # Run each agent
    for name, AgentCls, params in configs:
        # (re)instantiate environment to reset any agent-specific side effects
        env_agent = Environment(grid_fp=grid_fp, no_gui=True, sigma=sigma, random_seed=seed)
        _ = env_agent.reset()

        # Instantiate agent
        if AgentCls is ValueIterationAgent:
            agent = ValueIterationAgent()
        elif AgentCls is QLearningAgent:
            agent = QLearningAgent(
                grid_shape=env_agent.grid.shape,
                alpha=0.15, gamma=0.95,
                epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                seed=seed
            )
        else:
            agent = MCAgent(
                grid_shape=env_agent.grid.shape,
                gamma=1.0,
                epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                seed=seed
            )

        # Train & time
        t0 = time.time()
        if isinstance(agent, ValueIterationAgent):
            agent.train(env_agent)
        else:
            for ep in range(1, n_episodes + 1):
                state = env_agent.reset()
                done = False
                steps = 0
                while not done and steps < max_steps:
                    steps += 1
                    action = agent.take_action(state)
                    next_s, reward, done, info = env_agent.step(action)
                    agent.update(next_s, reward, info['actual_action'])
                    state = next_s
                agent.end_episode()
        t1 = time.time()
        training_time = t1 - t0

        # Evaluate greedy-run steps
        steps_to_goal, rewards = evaluate_steps(agent, env, max_steps)

        # Record results
        results.append({
            'grid': grid_fp.name,
            'agent': name,
            'training_time': training_time,
            'steps_to_goal': steps_to_goal,
            'rewards': rewards
        })

    return results


def main():
    # Grid configurations
    grids = [
        Path("grid_configs/1_goals_1.npy"),
        Path("grid_configs/2_goals_1.npy"),
        Path("grid_configs/3_goals_1.npy"),
        Path("grid_configs/1_goals_2.npy"),
        Path("grid_configs/2_goals_2.npy"),
        Path("grid_configs/3_goals_2.npy"),
        Path("grid_configs/1_goals_3.npy"),
        Path("grid_configs/2_goals_3.npy"),
        Path("grid_configs/3_goals_3.npy"),
        Path("grid_configs/1_goals_4.npy"),
        Path("grid_configs/2_goals_4.npy"),
        Path("grid_configs/3_goals_4.npy"),
        Path("grid_configs/1_goals_5.npy"),
        Path("grid_configs/2_goals_5.npy"),
        Path("grid_configs/3_goals_5.npy"),
    ]

    # Training parameters
    sigma = 0.1
    seed = 2025
    n_episodes = 5_000
    max_steps = 300

    summary = []
    for grid in grids:
        print(f"\n=== Training on {grid.name} ===")
        grid_results = train_on_grid(grid, sigma, seed, n_episodes, max_steps)
        for res in grid_results:
            print(f"{res['agent']:12} | Time: {res['training_time']:.2f}s | Steps: {res['steps_to_goal']} | Rewards: {res['rewards']}")
        summary.extend(grid_results + [{'grid': grid.name}])

    # Save summary
    Path("results").mkdir(exist_ok=True)
    np.save(Path("results") / "training_summary.npy", summary)
    print("\nAll trainings completed. Summary saved to results/training_summary.npy")

if __name__ == '__main__':
    main()
