"""Value Iteration Agent.

This agent uses the Value Iteration algorithm to compute the optimal policy.
"""
import numpy as np
from agents.base_agent import BaseAgent


class ValueIterationAgent(BaseAgent):
    """Agent that uses Value Iteration to determine the best action."""
    def __init__(self, env, gamma=0.9, theta=1e-6):
        """
        Args:
            env: The environment the agent interacts with.
            gamma: Discount factor for future rewards.
            theta: A small threshold for determining convergence.
        """
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_function = np.zeros(self.env.grid.size)  # Flattened value function
        self.policy = np.zeros(self.env.grid.size, dtype=int)  # Flattened policy
        self._value_iteration()

    def _state_to_index(self, state: tuple[int, int]) -> int:
        """Converts a 2D state (row, col) to a 1D index."""
        return state[0] * self.env.grid.shape[1] + state[1]

    def _index_to_state(self, index: int) -> tuple[int, int]:
        """Converts a 1D index to a 2D state (row, col)."""
        rows, cols = self.env.grid.shape
        return divmod(index, cols)

    def _value_iteration(self):
        """Perform the Value Iteration algorithm to compute the optimal policy."""
        num_states = self.env.grid.size
        while True:
            delta = 0
            for index in range(num_states):
                state = self._index_to_state(index)
                if self.env.grid[state] == 1:  # Skip walls
                    continue
                if self.env.grid[state] == 3:  # Terminal state
                    self.value_function[index] = 0
                    continue

                v = self.value_function[index]
                action_values = []
                for action in range(4):  # 4 possible actions: down, up, left, right
                    direction = self._action_to_direction(action)
                    next_state = (state[0] + direction[0], state[1] + direction[1])

                    # Check if next_state is valid
                    if not (0 <= next_state[0] < self.env.grid.shape[0] and
                            0 <= next_state[1] < self.env.grid.shape[1]):
                        continue

                    next_index = self._state_to_index(next_state)
                    reward = self.env._default_reward_function(self.env.grid, next_state)
                    action_value = reward + self.gamma * self.value_function[next_index]
                    action_values.append(action_value)

                if action_values:
                    self.value_function[index] = max(action_values)
                delta = max(delta, abs(v - self.value_function[index]))

            if delta < self.theta:
                break

        # Derive the policy from the value function
        for index in range(num_states):
            state = self._index_to_state(index)
            if self.env.grid[state] == 1:  # Skip walls
                continue
            if self.env.grid[state] == 3:  # Terminal state
                continue

            action_values = []
            for action in range(4):  # 4 possible actions: down, up, left, right
                direction = self._action_to_direction(action)
                next_state = (state[0] + direction[0], state[1] + direction[1])

                # Check if next_state is valid
                if not (0 <= next_state[0] < self.env.grid.shape[0] and
                        0 <= next_state[1] < self.env.grid.shape[1]):
                    continue

                next_index = self._state_to_index(next_state)
                reward = self.env._default_reward_function(self.env.grid, next_state)
                action_value = reward + self.gamma * self.value_function[next_index]
                action_values.append(action_value)

            if action_values:
                self.policy[index] = np.argmax(action_values)

    def _action_to_direction(self, action: int) -> tuple[int, int]:
        """Converts an action index to a direction (row_delta, col_delta)."""
        return {
            0: (1, 0),   # Down
            1: (-1, 0),  # Up
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }[action]

    def take_action(self, state: tuple[int, int]) -> int:
        """Take the action based on the computed policy.

        Args:
            state: The current state of the agent.

        Returns:
            The action to take.
        """
        state_index = self._state_to_index(state)
        return self.policy[state_index]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """No updates are needed for Value Iteration as it is computed offline."""
        pass