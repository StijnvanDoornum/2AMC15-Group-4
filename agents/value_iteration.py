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
    
    def _is_valid_state(self, pos):
        """Check if a position is within bounds and not a wall"""
        if not self._in_bounds(pos):
            return False
        x, y = pos
        return self.grid[x, y] not in (1, 2)  # Not a wall
    
    def _get_possible_actions(self, state):
        return ACTIONS  # All directions
    
    def train(self, env):
        self.grid = env.grid
        self.grid_shape = self.grid.shape
        self.V = np.zeros(self.grid_shape)
        self.policy = np.zeros(self.grid_shape, dtype=int)
        
        for it in range(self.max_iterations):
            delta = 0
            
            for x in range(self.grid_shape[0]):
                for y in range(self.grid_shape[1]):
                    if self.grid[x, y] in (1, 2):  # Skip walls
                        continue
                    
                    state = (x, y)
                    current_v = self.V[x, y]
                    max_value = float('-inf')
                    best_action = 0
                    
                    # For each possible action from this state
                    for a in self._get_possible_actions(state):
                        value = 0
                        
                        # For each possible outcome of this action (stochastic transitions)
                        for b in ACTIONS:
                            # Probability of action b occurring when intending to take action a
                            prob = 1 - env.sigma if a == b else env.sigma / 3
                            
                            dx, dy = ACTION_TO_DELTA[b]
                            next_pos = (x + dx, y + dy)
                            
                            # If next position is out of bounds or a wall, stay in place
                            if not self._is_valid_state(next_pos):
                                next_pos = state
                            
                            next_x, next_y = next_pos
                            # Get reward for this transition
                            reward = env.reward_fn(self.grid, next_pos)
                            
                            # Update value using Bellman equation
                            value += prob * (reward + self.gamma * self.V[next_x, next_y])
                        
                        if value > max_value:
                            max_value = value
                            best_action = a
                    
                    # Update the value and policy
                    self.V[x, y] = max_value
                    self.policy[x, y] = best_action
                    
                    # Track the maximum change in value
                    delta = max(delta, abs(current_v - max_value))
            
            # Check for convergence
            if delta < self.theta:
                print(f"Converged after {it+1} iterations")
                break
        
        # Print debug information
        print("----------------------------------------------------------------------")
        
        # Trace optimal path if start state is available
        start_state = getattr(self, "_start_state", None)
        if start_state is None:
            print("Start state not provided; cannot trace optimal path.")
            return
            
        state = start_state
        visited = set()
        print("\nOptimal Path State Values:")
        
        while True:
            state_int = (int(state[0]), int(state[1]))
            if state_int in visited:
                print("Loop detected — stopping trace.")
                break
                
            visited.add(state_int)
            print(f"State: {state_int}, Value: {self.V[state_int]}")
            
            action = self.policy[state_int]
            dx, dy = ACTION_TO_DELTA[action]
            next_state = (state[0] + dx, state[1] + dy)
            
            if not self._is_valid_state(next_state):
                next_state = state  # Stay in place if invalid
                
            state = next_state
            
            # Check if we've reached a terminal state (this depends on your environment)
            # You might need to adjust this condition based on your specific implementation
            if self.grid[int(state[0]), int(state[1])] == 2:  # Assuming 2 is a terminal state
                print(f"Terminal state reached: {state}, Value: {self.V[int(state[0]), int(state[1])]}")
                break
                
        print("------------------------------------------------------")
    
    def take_action(self, state):
        if self.policy is None:
            raise ValueError("Policy not computed. Run train(env) first.")
        x, y = state
        return self.policy[x, y]
    
    def update(self, state, reward, action):
        pass  # Value iteration is offline, no need to update per step