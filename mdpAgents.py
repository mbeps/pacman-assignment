from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):
    """
    An agent that uses value iteration to compute optimal actions.
    All decision making is done through the MDP framework.
    """
    def __init__(self):
        # Grid representation
        self.grid = None
        self.utilities = None
        self.rewards = None
        self.width = None 
        self.height = None
        
        # MDP parameters
        self.discount = 0.7        # Regular discount factor
        self.danger_discount = 0.95  # Slower decay in danger zones
        self.living_reward = -0.04
        self.food_reward = 10
        self.ghost_reward = -1000
        self.danger_radius = 4       # Size of danger zone around ghosts
        self.iterations = 100
        self.convergence_threshold = 0.01

    def registerInitialState(self, state):
        """Initialize the agent with the game state"""
        # Get grid dimensions from corners
        corners = api.corners(state)
        self.width = max(x for x, y in corners) + 1
        self.height = max(y for x, y in corners) + 1
        
        # Initialize grids for utilities and rewards
        self.utilities = self.create_grid(0.0)
        self.rewards = self.create_grid(self.living_reward)
        
        # Mark walls
        walls = api.walls(state)
        for x, y in walls:
            self.rewards[x][y] = None
            self.utilities[x][y] = None
            
        # Run initial state setup and value iteration
        self.update_state(state)

    def create_grid(self, initial_value):
        """Create a width x height grid with initial_value"""
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def update_state(self, state):
        """Update the MDP state and rewards based on current game state"""
        # Reset non-wall states to living reward
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward
        
        # Incorporate all state elements into the MDP's reward function
        self.update_rewards(state)
        
        # Run value iteration to compute utilities
        self.value_iteration()

    def update_rewards(self, state):
        """Update rewards based on current state"""
        self.current_state = state  # Store current state for danger zone checking
        
        # Reset non-wall states to living reward
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward

        # Process ghost states
        ghost_states = api.ghostStates(state)
        for (ghost_x, ghost_y), scared in ghost_states:
            if not scared:
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                
                # Apply exponential decay penalties in the danger zone
                for x in range(self.width):
                    for y in range(self.height):
                        if self.rewards[x][y] is not None:
                            manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
                            if manhattan_dist <= self.danger_radius:
                                # Using 0.8 as base means:
                                # distance 0: penalty = ghost_reward * 1     = -1000
                                # distance 1: penalty = ghost_reward * 0.8   = -800
                                # distance 2: penalty = ghost_reward * 0.64  = -640
                                # distance 3: penalty = ghost_reward * 0.512 = -512
                                # distance 4: penalty = ghost_reward * 0.409 = -409
                                penalty = self.ghost_reward * (0.8 ** manhattan_dist)
                                self.rewards[x][y] = min(self.rewards[x][y], penalty)

        # Add food rewards after ghost penalties
        for x, y in api.food(state):
            self.rewards[x][y] += self.food_reward

    def value_iteration(self):
        """Perform value iteration to compute utilities for all states"""
        for _ in range(self.iterations):
            new_utilities = self.create_grid(0.0)
            max_change = 0.0
            
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        utility = self.compute_state_utility(x, y)
                        new_utilities[x][y] = utility
                        change = abs(utility - self.utilities[x][y])
                        max_change = max(max_change, change)
            
            self.utilities = new_utilities
            
            if max_change < self.convergence_threshold:
                break

    def compute_state_utility(self, x, y):
        """Compute utility for a state using the Bellman equation with different discount rates"""
        if self.rewards[x][y] is None:  # Wall
            return None
            
        R = self.rewards[x][y]
        
        # Check if we're in a danger zone
        in_danger_zone = False
        ghost_states = api.ghostStates(self.current_state)
        for (ghost_x, ghost_y), scared in ghost_states:
            if not scared:
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
                if manhattan_dist <= self.danger_radius:
                    in_danger_zone = True
                    break
        
        # Use appropriate discount factor
        discount = self.danger_discount if in_danger_zone else self.discount
        
        # Get maximum expected utility over all actions
        max_utility = float("-inf")
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            exp_utility = self.get_expected_utility(x, y, action)
            max_utility = max(max_utility, exp_utility)
            
        return R + discount * max_utility

    def get_expected_utility(self, x, y, action):
        """Compute expected utility of taking an action in state (x,y)"""
        successors = self.get_successor_states(x, y, action)
        
        exp_utility = 0.0
        for (next_x, next_y), prob in successors.items():
            if (0 <= next_x < self.width and 
                0 <= next_y < self.height and 
                self.utilities[next_x][next_y] is not None):
                exp_utility += prob * self.utilities[next_x][next_y]
            else:
                # If would hit wall or go out of bounds, stay in same place
                exp_utility += prob * self.utilities[x][y]
            
        return exp_utility

    def get_successor_states(self, x, y, action):
        """Return dictionary of successor states and their probabilities"""
        successors = {}
        
        # Get direction vectors
        if action == Directions.NORTH:
            intended = (0, 1)
            perpendicular = [(1, 0), (-1, 0)]
        elif action == Directions.SOUTH:
            intended = (0, -1)
            perpendicular = [(1, 0), (-1, 0)]
        elif action == Directions.EAST:
            intended = (1, 0)
            perpendicular = [(0, 1), (0, -1)]
        elif action == Directions.WEST:
            intended = (-1, 0)
            perpendicular = [(0, 1), (0, -1)]
            
        # Add intended direction (0.8 probability)
        next_x = x + intended[0]
        next_y = y + intended[1]
        successors[(next_x, next_y)] = 0.8
        
        # Add perpendicular directions (0.1 probability each)
        for dx, dy in perpendicular:
            next_x = x + dx
            next_y = y + dy
            successors[(next_x, next_y)] = 0.1
            
        return successors

    def getAction(self, state):
        """Get the optimal action using maximum expected utility"""
        # Update MDP state and recompute utilities
        self.update_state(state)
        
        # Get current position and legal actions
        x, y = api.whereAmI(state)
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Find action with maximum expected utility
        max_utility = float("-inf")
        best_action = None
        
        for action in legal:
            exp_utility = self.get_expected_utility(x, y, action)
            if exp_utility > max_utility:
                max_utility = exp_utility
                best_action = action
                
        # If no legal actions or all have same utility, choose random
        if best_action is None and legal:
            best_action = random.choice(legal)
            
        return api.makeMove(best_action, legal)

    def final(self, state):
        """Called at the end of each game"""
        # Reset any necessary state variables here
        self.current_state = None