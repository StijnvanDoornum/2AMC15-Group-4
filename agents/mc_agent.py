"""
Monte Carlo agent for the DIC delivery‑robot environment.

"""
from __future__ import annotations
import numpy as np
from agents.base_agent import BaseAgent



class MCAgent(BaseAgent):
    """

    """

    def __init__(
        self,
        grid_shape: tuple[int, int],     # from env.grid.shape
        gamma: float = 0.95,             # the discount factor
        epsilon: float = 1.0,            # start of epsilon‑greedy schedule
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = 0,
    ):
        self.rng = np.random.default_rng(seed) 
        self.n_cols, self.n_rows = grid_shape          # environment returns (cols, rows)
        self.n_actions = 4                             # up,down,left,right
        self.Q = self.rng.random((self.n_rows, self.n_cols, self.n_actions), dtype=np.float32)
        self.returns = np.empty((self.n_rows, self.n_cols, self.n_actions), dtype=object)
        for index, _ in np.ndenumerate(self.returns):
            self.returns[index] = []  # each element gets a unique list

        self.gamma = gamma

        self.epsilon = float(epsilon)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.episodic_state_action_pairs = []
        self._last_state = None
        self._last_action = None

    def take_action(self, state: tuple[int, int]) -> int:
        row, col = state
        self._last_state = state

        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
        else:
            action = int(np.argmax(self.Q[row, col]))
        self._last_action = action
        return action

    # This is called by the training loop AFTER env.step
    def update(self, state: tuple[int, int], reward: float, executed_action: int):
        self.episodic_state_action_pairs.append((self._last_state, executed_action, reward))

    # Can be called to end the episode
    def end_episode(self):
        # update the Q value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        G = 0
        visited = set()
        total_returns = {}
        for state, action, reward in reversed(self.episodic_state_action_pairs):
            row, col = state
            G = reward + self.gamma * G
            if (row, col, action) not in total_returns:
                total_returns[row, col, action] = []
            total_returns[row, col, action].append(G)
            visited.add((state, action))

        # Calculate the weighted mean of the returns for each state-action pair depended on the number of times it was visited
        for state, action in visited:
            row, col = state
            if len(total_returns[row, col, action]) > 0:
                self.returns[row, col, action].append(np.mean(total_returns[row, col, action]))


        # Flatten the updates to reduce nested loops
        for (state, action) in visited:
            row, col = state
            self.Q[row, col, action] = np.mean(self.returns[row, col, action])

        self.episodic_state_action_pairs = []  # reset the episodic state-action pairs for the next episode
        self._last_state = None