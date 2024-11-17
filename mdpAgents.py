# mdpAgents.py
# parsons/20-nov-2017
#
# Version 3.0
#
# An MDP-based agent that handles ghosts using value iteration.

from pacman import Directions
from game import Agent
import api
import random
import game
import util
from copy import deepcopy

class MDPAgent(Agent):
    """
    An agent that uses value iteration to compute optimal actions.
    All decision making is done through the MDP framework with enhanced ghost handling.
    """
    def __init__(self):
        # Grid representation
        self.grid = None
        self.utilities = None
        self.rewards = None
        self.width = None 
        self.height = None
        
        # MDP parameters
        self.discount = 0.9
        self.living_reward = -0.04
        self.food_reward = 10
        self.ghost_penalty = -50  # Base penalty for ghost square
        self.scared_ghost_reward = 20  # Reward for eating scared ghost
        self.iterations = 100
        self.convergence_threshold = 0.01
        
        # Ghost parameters
        self.ghost_radius = 2  # How far ghost danger spreads
        self.ghost_discount = 0.7  # How quickly ghost danger decreases with distance

    def registerInitialState(self, state):
        """Initialize the agent with the game state"""
        corners = api.corners(state)
        self.width = max(x for x, y in corners) + 1
        self.height = max(y for x, y in corners) + 1
        
        self.utilities = self.create_grid(0.0)
        self.rewards = self.create_grid(self.living_reward)
        
        walls = api.walls(state)
        for x, y in walls:
            self.rewards[x][y] = None
            self.utilities[x][y] = None
            
        self.update_state(state)

    def create_grid(self, initial_value):
        """Create a width x height grid with initial_value"""
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_state(self, state):
        """Update the MDP state and rewards based on current game state"""
        # Reset non-wall states to living reward
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward
        
        self.update_rewards(state)
        self.value_iteration()

    def update_rewards(self, state):
        """Update rewards based on current state with enhanced ghost handling"""
        # Food rewards
        for x, y in api.food(state):
            self.rewards[x][y] += self.food_reward
            
        # Ghost rewards with danger zone
        ghost_states = api.ghostStates(state)
        pacman_pos = api.whereAmI(state)
        
        for (ghost_x, ghost_y), scared in ghost_states:
            ghost_x = int(ghost_x)
            ghost_y = int(ghost_y)
            
            # Handle each position within ghost radius
            for x in range(max(0, ghost_x - self.ghost_radius), 
                         min(self.width, ghost_x + self.ghost_radius + 1)):
                for y in range(max(0, ghost_y - self.ghost_radius),
                             min(self.height, ghost_y + self.ghost_radius + 1)):
                    
                    if self.rewards[x][y] is not None:  # If not a wall
                        distance = self.manhattan_distance((ghost_x, ghost_y), (x, y))
                        
                        if distance <= self.ghost_radius:
                            if scared:
                                # Scared ghosts are attractive, but reward decreases with distance
                                reward = self.scared_ghost_reward * (self.ghost_discount ** distance)
                                self.rewards[x][y] += reward
                            else:
                                # Normal ghosts create danger zones that increase with proximity
                                penalty = self.ghost_penalty * (self.ghost_discount ** distance)
                                # Direct ghost square is extremely dangerous
                                if distance == 0:
                                    penalty *= 2
                                self.rewards[x][y] += penalty

    def get_transition_prob(self, curr_pos, next_pos, action):
        """
        Get transition probability for moving from curr_pos to next_pos with given action
        Accounts for walls by returning 0 probability for wall transitions
        """
        x, y = curr_pos
        next_x, next_y = next_pos
        
        # Check if next position is a wall or out of bounds
        if (next_x < 0 or next_x >= self.width or 
            next_y < 0 or next_y >= self.height or 
            self.rewards[next_x][next_y] is None):
            return 0.0
            
        # Determine intended and perpendicular directions
        if action == Directions.NORTH:
            if next_y == y + 1 and next_x == x:
                return 0.8  # Intended direction
            if (next_y == y and (next_x == x + 1 or next_x == x - 1)):
                return 0.1  # Perpendicular directions
        elif action == Directions.SOUTH:
            if next_y == y - 1 and next_x == x:
                return 0.8
            if (next_y == y and (next_x == x + 1 or next_x == x - 1)):
                return 0.1
        elif action == Directions.EAST:
            if next_x == x + 1 and next_y == y:
                return 0.8
            if (next_x == x and (next_y == y + 1 or next_y == y - 1)):
                return 0.1
        elif action == Directions.WEST:
            if next_x == x - 1 and next_y == y:
                return 0.8
            if (next_x == x and (next_y == y + 1 or next_y == y - 1)):
                return 0.1
                
        return 0.0

    def get_expected_utility(self, x, y, action):
        """
        Compute expected utility of taking an action in state (x,y)
        using the non-deterministic Bellman equation
        """
        if self.rewards[x][y] is None:  # Wall
            return float('-inf')
            
        exp_utility = 0.0
        
        # Consider all possible next states
        for next_x in range(max(0, x-1), min(self.width, x+2)):
            for next_y in range(max(0, y-1), min(self.height, y+2)):
                # Get transition probability
                prob = self.get_transition_prob((x, y), (next_x, next_y), action)
                
                if prob > 0:
                    # If transition leads to wall, stay in current state
                    if (self.rewards[next_x][next_y] is None):
                        exp_utility += prob * self.utilities[x][y]
                    else:
                        exp_utility += prob * self.utilities[next_x][next_y]
        
        return exp_utility

    def value_iteration(self):
        """Perform value iteration to compute utilities for all states"""
        for _ in range(self.iterations):
            new_utilities = self.create_grid(0.0)
            max_change = 0.0
            
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        max_utility = float('-inf')
                        
                        # Calculate maximum expected utility over all actions
                        for action in [Directions.NORTH, Directions.SOUTH, 
                                     Directions.EAST, Directions.WEST]:
                            exp_utility = self.get_expected_utility(x, y, action)
                            max_utility = max(max_utility, exp_utility)
                        
                        # Bellman update
                        new_utilities[x][y] = self.rewards[x][y] + self.discount * max_utility
                        
                        # Track maximum change
                        change = abs(new_utilities[x][y] - self.utilities[x][y])
                        max_change = max(max_change, change)
                    else:
                        new_utilities[x][y] = None
            
            self.utilities = new_utilities
            
            # Check for convergence
            if max_change < self.convergence_threshold:
                break

    def getAction(self, state):
        """Get the optimal action using maximum expected utility"""
        self.update_state(state)
        
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
                
        if best_action is None and legal:
            best_action = random.choice(legal)
            
        return api.makeMove(best_action, legal)

    def final(self, state):
        """Called at the end of each game"""
        # Reset state variables
        self.utilities = None
        self.rewards = None