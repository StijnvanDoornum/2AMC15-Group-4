from __future__ import annotations
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from world.environment import Environment
from agents.mc_agent import MCAgent


# configuration
GRID_PATH         = Path("grid_configs/A1_grid.npy")  # picking the grid
SIGMA             = 0.5 #0.30    # environment stochasticity
N_EPISODES        = 40_000   # total training episodes
MAX_STEPS         = 350     # safety cap per episode
VIZ_INTERVAL      = 500     # run a GUI episode every … episodes
MOVING_AVG_WINDOW = 1000      # size of moving‑average window for the plot

# visualizing an episode using the GUI
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


# setting up the environment
train_env = Environment(GRID_PATH, no_gui=True, sigma=SIGMA, random_seed=2025)
state     = train_env.reset()

agent = MCAgent(
    grid_shape=train_env.grid.shape,
    gamma=1,
    epsilon=1, epsilon_min=0, epsilon_decay=0.999,
    seed=2025
)

episode_returns: list[float] = []

# training loop
for ep in tqdm(range(1, N_EPISODES + 1), desc="Training", ncols=100):
    state = train_env.reset()
    done  = False
    G     = 0
    step  = 0

    while not done and step < MAX_STEPS:
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
            f"MovingAvg {ma:>6.1f} | ε {agent.epsilon:5.3f} | MAX_STEPS {MAX_STEPS:>4} | "
        )

    # show with GUI
    if ep % VIZ_INTERVAL == 0:
        print(f"\n Visualising policy after episode {ep} …")
        run_gui_episode(agent, GRID_PATH, SIGMA, MAX_STEPS)
        print("Training continues …\n")

plt.figure(figsize=(8, 4))
plt.plot(episode_returns, alpha=0.3, label="episode return")
if len(episode_returns) >= MOVING_AVG_WINDOW:
    ma = np.convolve(
        episode_returns,
        np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
        mode="valid"
    )
    plt.plot(
        range(MOVING_AVG_WINDOW - 1, N_EPISODES),
        ma,
        label=f"{MOVING_AVG_WINDOW}-episode moving average"
    )
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Monte Carlo learning curve")
plt.legend()
plt.tight_layout()
plt.show()

timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
Path("results").mkdir(exist_ok=True)
np.save(Path("results") / f"q_table_{timestamp}.npy", agent.Q)
print(f"\nTraining finished – Q‑table saved as results/q_table_{timestamp}.npy")
mean_return = np.mean(episode_returns)
print(f"Mean return over {N_EPISODES} episodes (σ={SIGMA}): {mean_return:.2f}")
