"""
Q‑learning agent for the DIC delivery‑robot environment.
Author: Jord & Stijn
"""
from __future__ import annotations
import numpy as np
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular ε‑greedy Q‑learning.
    Works on the raw (row, col) state that the environment returns.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],     # from env.grid.shape
        alpha: float = 0.15,             # the learning‑rate
        gamma: float = 0.95,             # the discount factor
        epsilon: float = 1.0,            # start of epsilon‑greedy schedule
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = 0,
    ):
        self.n_cols, self.n_rows = grid_shape          # environment returns (cols, rows)
        self.n_actions = 4                             # up,down,left,right
        self.Q = np.zeros((self.n_rows, self.n_cols, self.n_actions), dtype=np.float32)

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = float(epsilon)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # this if for internal bookkeeping
        self._last_state: tuple[int, int] | None = None

    def take_action(self, state: tuple[int, int]) -> int:
        row, col = state
        self._last_state = state

        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(self.n_actions))
        else:
            action = int(np.argmax(self.Q[row, col]))

        return action

    # This is called by the training loop AFTER env.step
    def update(self, state: tuple[int, int], reward: float, executed_action: int):
        """Update Q(s,a) using the action that *actually* happened inside the
        environment (env.info['actual_action'])."""
        if self._last_state is None:
            return  # this is the first step of the episode

        s_row, s_col = self._last_state
        a = executed_action                               # This can differ from chosen action!!
        next_row, next_col = state

        td_target = reward + self.gamma * np.max(self.Q[next_row, next_col])
        td_error = td_target - self.Q[s_row, s_col, a]
        self.Q[s_row, s_col, a] += self.alpha * td_error

    # Can be called to end the episode
    def end_episode(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
