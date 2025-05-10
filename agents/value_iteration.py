import numpy as np
from agents.base_agent import BaseAgent

ACTIONS = [0, 1, 2, 3]  
ACTION_TO_DELTA = {
    0: (0, 1),
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0)
}

class ValueIterationAgent(BaseAgent):
    def __init__(self, gamma=0.9, theta=1e-4, max_iterations=1000):
        super().__init__()
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.V = None
        self.policy = None
        self.grid_shape = None
        self.grid = None

    def _in_bounds(self, pos):
        x, y = pos
        return (
            0 <= x and x < self.grid_shape[0] and
            0 <= y and y < self.grid_shape[1]
        )
    def _get_possible_actions(self, state):
        return ACTIONS  # All directions 

    def train(self, env):
        self.grid = env.grid
        self.grid_shape = self.grid.shape
        self.V = np.zeros(self.grid_shape)
        self.policy = np.zeros(self.grid_shape, dtype=int)

        for it in range(self.max_iterations):
            delta = 0
            new_V = np.copy(self.V)

            for x in range(self.grid_shape[0]):
                for y in range(self.grid_shape[1]):
                    if self.grid[x, y] in (1, 2):  # Skip walls
                        continue
                    state = (x, y)
                    max_value = float('-inf')
                    best_action = 0

                    for a in self._get_possible_actions(state):
                        value = 0
                        for b in ACTIONS:
                            prob = 1 - env.sigma if a == b else env.sigma / 3  # probability of action b occurring when you intended to take action a
                            dx, dy = ACTION_TO_DELTA[b] #movement direction for action b 
                            next_state = (x + dx, y + dy)
                            if not self._in_bounds(next_state):
                                next_state = state  # Bounce back
                            reward = env.reward_fn(self.grid, next_state)
                            value += prob * (reward + self.gamma * self.V[next_state])
                        
                        if value > max_value:
                            max_value = value
                            best_action = a

                    new_V[state] = max_value
                    self.policy[state] = best_action
                    delta = max(delta, abs(self.V[state] - max_value))

            self.V = new_V
            if delta < self.theta:
                break

        print("----------------------------------------------------------------------")
        #printing number of iterations until optimal
        # print("Number of iterations = ", it)

        #printing the values for all states visited in the optimal path
        start_state = getattr(self, "_start_state", None)
        if start_state is None:
            print("Start state not provided; cannot trace optimal path.")
            return

        state = start_state
        visited = set()

        print("\nOptimal Path State Values:")
        while True:
            if state in visited:
                print("Loop detected — stopping trace.")
                break
            visited.add(state)

            state_int = (int(state[0]), int(state[1]))
            print(f"State: {state_int}, Value: {self.V[state]}")


            action = self.policy[state]
            dx, dy = ACTION_TO_DELTA[action]
            next_state = (state[0] + dx, state[1] + dy)

            if not self._in_bounds(next_state) or self.grid[next_state] in (1, 2):
                break

            state = next_state
        print("------------------------------------------------------")
        


    def take_action(self, state):
        if self.policy is None:
            raise ValueError("Policy not computed. Run train(env) first.")
        return self.policy[state]
    

    def update(self, state, reward, action):
        pass  # Value iteration is offline, no need to update per step