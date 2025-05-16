"""
Train your RL Agent in this file. 
"""

from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import time
import numpy as np

try:
    from world import Environment
    from agents.value_iteration import ValueIterationAgent
    
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys
    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )
    if root_path not in sys.path:
        sys.path.extend(root_path)
    from world import Environment
    from agents.random_agent import RandomAgent

def parse_args():
    p = ArgumentParser(description="DIC Reinforcement Learning Trainer.")
    p.add_argument("GRID", type=Path, nargs="+",
                   help="Paths to the grid file to use. There can be more than "
                        "one.")
    p.add_argument("--no_gui", action="store_true",
                   help="Disables rendering to train faster")
    p.add_argument("--sigma", type=float, default=0.1,
                   help="Sigma value for the stochasticity of the environment.")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second to render at. Only used if "
                        "no_gui is not set.")
    p.add_argument("--iter", type=int, default=1000,
                   help="Number of iterations to go through.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Random seed value for the environment.")
    return p.parse_args()


def main(grid_paths: list[Path], no_gui: bool, iters: int, fps: int,
         sigma: float, random_seed: int):
    """Main loop of the program."""

    for grid in grid_paths:
        start_time = time.time()
        
        # Set up the environment
        env = Environment(grid, no_gui,sigma=sigma, target_fps=fps, 
                          random_seed=random_seed)
        
        # Initialize agent
        # agent = RandomAgent()
        
        
        
        # Always reset the environment to initial state
        state = env.reset()
        agent = ValueIterationAgent()
        agent._start_state = state
        agent.train(env)
        evaluate_mean_return(agent, grid, sigma=sigma, episodes=20000, max_steps=600)

        # for _ in trange(iters):
            
        #     # Agent takes an action based on the latest observation and info.
        #     action = agent.take_action(state)

        #     # The action is performed in the environment
        #     state, reward, terminated, info = env.step(action)
            
        #     # If the final state is reached, stop.
        #     if terminated:
        #         break

        #     agent.update(state, reward, info["actual_action"])

        # # Evaluate the agent
        # Environment.evaluate_agent(grid, agent, iters, sigma, random_seed=random_seed)
        # end_time = time.time()
        # print(f"Training + evaluation on {grid.name} completed in {end_time - start_time:.2f} seconds.")

def evaluate_mean_return(agent, grid_fp, sigma, episodes=20000, max_steps=600, seed=2025):
    returns = []

    for ep in range(episodes):
        env = Environment(
            grid_fp=grid_fp,
            no_gui=True,
            sigma=sigma,
            agent_start_pos=None,
            target_fps=-1,
            random_seed=seed + ep  # vary seed for randomness
        )
        state = env.reset()
        G = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = agent.take_action(state)
            state, reward, done, info = env.step(action)
            G += reward
            steps += 1

        returns.append(G)

    mean_return = np.mean(returns)
    print(f"Mean return over {episodes} episodes (σ={sigma}): {mean_return:.2f}")
    return mean_return


if __name__ == '__main__':
    args = parse_args()
    main(args.GRID, args.no_gui, args.iter, args.fps, args.sigma, args.random_seed)
