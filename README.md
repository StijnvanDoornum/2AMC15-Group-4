## Available Agents

There are three main agents you can work with in this environment:

### 1. Value Iteration Agent

- Use the file: `train_VI.py`
- Example commands:
  ```bash
  python train_VI.py grid_configs/A1_grid.npy --sigma 0.0 --fps 5
  ```
  or for the larger grid:
  ```bash
  python train_VI.py grid_configs/hardex_grid.npy --sigma 0.0 --fps 5
  ```
- The `--fps` flag controls the speed of the GUI so you can observe the agent's movement.

### 2. Q-Learning Agent

- Use the file: `train_q_learning.py`
- You can modify the grid path, number of episodes, and other parameters at the beginning of the file, or pass them as command-line arguments.
- Example command:
  ```bash
  python train_q_learning.py
  ```
- Example with custom parameters:
  ```bash
  python train_q_learning.py --alpha 0.01 --gamma 0.99 --epsilon 0.5 --episodes 5000 --max_steps 800
  ```
- To run a grid search for Q-learning using experiment.py:
  ```bash
  python experiment.py --experiment hyperparameter_comparison --agent q_learning
  ```

### 3. Monte Carlo Agent

- Use the file: `train_mc.py`
- You can modify the grid path, number of episodes, and other parameters at the beginning of the file, or pass them as command-line arguments.
- Example command:
  ```bash
  python train_mc.py
  ```
- Example with custom parameters:
  ```bash
  python train_mc.py --gamma 0.99 --epsilon 0.5 --epsilon_decay 0.995 --episodes 20000 --max_steps 500
  ```
- To run a grid search for Monte Carlo using experiment.py:
  ```bash
  python experiment.py --experiment hyperparameter_comparison --agent monte_carlo
  ```

> **Note:**  
> If you encounter errors about not finding the agent, check the import statements in the training files and ensure they match your folder and file structure.

### Additional Experimentation

You can experiment with different values of `sigma` for value iteration and plot the resulting rewards using the `sigma_testing.py` script:
```bash
python sigma_testing.py
```

## Parameter Configuration for Q-Learning and Monte Carlo

Below are the main parameters and grid search values used for the Q-Learning and Monte Carlo agents in this environment. These are set in `experiment.py` for systematic experimentation.

### Q-Learning Agent
- **Learning Rate (alpha):** `[0.05, 0.10, 0.15, 0.20]`
- **Discount Factor (gamma):** `[0.60, 0.80, 0.90, 0.99]`
- **Epsilon (initial):** `1.0` (for epsilon-greedy exploration)
- **Epsilon Decay:** `0.995`
- **Epsilon Min:** `0.05`
- **Sigma (stochasticity):** `0.1`
- **Episodes:** `3000`
- **Max Steps per Episode:** `600`

#### Reward Function Grid (for reward_comparison):
- **Empty Tile Reward:** `[-0.1, -1]`
- **Wall/Obstacle Reward:** `[-1, -5, -10, -15]`
- **Target Reward:** `[5, 10, 20]`

### Monte Carlo Agent
- **Discount Factor (gamma):** `1.0`
- **Epsilon (initial):** `1.0` (for epsilon-greedy exploration)
- **Epsilon Decay:** `[0.999, 0.9995, 0.9999]` (for hyperparameter grid)
- **Epsilon Min:** `0.05` (or `0.00` for some experiments)
- **Episodes:** `10000` (for hyperparameter grid)
- **Max Steps per Episode:** `[250, 350, 500]` (for hyperparameter grid)

#### Reward Function Grid (for reward_comparison):
- **Empty Tile Reward:** `[-0.1, -1]`
- **Wall/Obstacle Reward:** `[-1, -5, -10, -15]`
- **Target Reward:** `[5, 10, 20]`

> **Note:**
> The above values are used for grid search and systematic comparison in `experiment.py`. You can modify these in the script to explore other configurations.

## The `experiment.py` File

The `experiment.py` file is a comprehensive script for running and comparing a wide range of experiments with different agents, hyperparameters, and reward functions. It is designed to facilitate systematic evaluation and visualization of RL agent performance in the DIC environment.

### Usage

You can run the experiment file with various arguments to select the type of experiment and agent:

```bash
python experiment.py --experiment <experiment_type> --agent <agent_type>
```

#### Arguments

- `--experiment`  
  Specifies the type of experiment to run. Available options include:
  - `reward_comparison` — Grid search over different reward functions.
  - `agent_comparison` — Compare Q-learning, Monte Carlo, and Value Iteration agents.
  - `hyperparameter_comparison` — Grid search over agent hyperparameters (e.g., learning rate, gamma).
  - `agent_comparison_with_optimal_rewards` — Compare all agents using their optimal reward parameters.
  - `epsilon_comparison` — Grid search over epsilon-greedy exploration parameters.
  - `optimal_epsilon_comparison` — Compare Q-learning and Monte Carlo with their best epsilon settings.
  - `reward_grid_comparison` — Compare a single agent on several hand-picked reward parameter sets.
  - `q_mc_selected_epsilon` — Compare Q-learning and Monte Carlo on selected epsilon schedules in a two-panel plot.

- `--agent`  
  Specifies the agent to use for the experiment (when relevant). Options:
  - `q_learning`
  - `monte_carlo`
  - `value_iteration`

### Example Commands

To run a reward function grid search for Q-learning:
```bash
python experiment.py --experiment reward_comparison --agent q_learning
```

To run a hyperparameter grid search for Q-learning:
```bash
python experiment.py --experiment hyperparameter_comparison --agent q_learning
```

You can type `python experiment.py -h` to see all available options and arguments for running experiments.

### Output

All experiments generate plots and save them in the `figs/` directory for easy analysis and inclusion in reports.