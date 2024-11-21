from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):
    def __init__(self):
        # Grid representation
        self.grid = None
        self.utilities = None
        self.rewards = None
        self.width = None 
        self.height = None
        
        # MDP parameters
        self.discount = 0.75  
        self.danger_discount = 0.95
        self.living_reward = -0.04
        self.food_reward = 20 
        self.ghost_reward = -1000
        
        # These will be set based on map size
        self.danger_radius = None
        self.food_radius = None
        
        self.iterations = 1500
        self.convergence_threshold = 0.001

    def registerInitialState(self, state):
        """
        Initialises the MDP agent a new game starts.
        Its sets up:
        - The grid representation of the game
        - The dimensions of the grid
        - The danger and food radius based on the number of ghosts
        - The utility and reward grids
        - The walls of the game
        - Updates the state representation

        Parameters:
        state (GameState): The current game state
        """
        corners = api.corners(state)
        self.width = max(x for x, y in corners) + 1
        self.height = max(y for x, y in corners) + 1
        
        ghost_count = len(api.ghosts(state))
        
        if ghost_count == 2:
            self.danger_radius = 4
            self.food_radius = 3
        elif ghost_count == 1:  
            self.danger_radius = 4
            self.food_radius = 3
        
        self.utilities = self.create_grid(0.0)
        self.rewards = self.create_grid(self.living_reward)
        
        walls = api.walls(state)
        for x, y in walls:
            self.rewards[x][y] = None
            self.utilities[x][y] = None
            
        self.update_state(state)

    def create_grid(self, initial_value):
        """
        Creates a 2D grid with the given initial value.

        Parameters:
        initial_value (float): The initial value to fill the grid with

        Returns:
        list: A 2D list representing
        """
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def update_state(self, state):
        """
        Updates the internal state representation of the agent.
        This maintains a fresh state representation ensuring the utility values are up-to-date.
        This is called when the game starts, after each move and state changes. 

        Works by:
        - Resetting all non-wall states to the living reward (walls are None)
        - Updates the rewards based on the current state (food and ghost rewards)
        - Propagates influences across the map
        - Runs value iteration to compute utilities for each state

        Parameters:
        state (GameState): The current game state
        """
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward
        
        self.update_rewards(state)
        self.value_iteration()

    def update_rewards(self, state):
        """
        Updates the rewards grid based on the current state.
        - Resets all non-wall states to the living reward   
        - Applies penalties for ghost states based on distance
        - Applies rewards for food states based on distance
        - Propagates food rewards across the map

        Uses distance as weight to apply penalties and rewards when propagating.
        This insures that the influence does not decay too quickly and is more localized.
        However, the decay should not remain large for larger distances to avoid overfitting.

        Parameters:
        state (GameState): The current game state
        """
        self.current_state = state
        
        # Reset non-wall states to living reward
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward

        # Process ghost states first (stronger influence)
        ghost_states = api.ghostStates(state)
        for (ghost_x, ghost_y), scared in ghost_states:
            if not scared:
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                
                for x in range(self.width):
                    for y in range(self.height):
                        if self.rewards[x][y] is not None:
                            manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
                            if manhattan_dist <= self.danger_radius:
                                # Using 0.7 as base for faster decay
                                penalty = self.ghost_reward * (0.7 ** manhattan_dist)
                                self.rewards[x][y] = min(self.rewards[x][y], penalty)

        # Process food rewards with propagation
        food_locations = api.food(state)
        for food_x, food_y in food_locations:
            self.rewards[food_x][food_y] += self.food_reward
            
            # Propagate food reward with fast decay
            for x in range(self.width):
                for y in range(self.height):
                    if self.rewards[x][y] is not None:
                        manhattan_dist = abs(x - food_x) + abs(y - food_y)
                        if manhattan_dist <= self.food_radius and manhattan_dist > 0:
                            # Using 0.5 as base for very fast decay
                            # This ensures food influence drops off quickly
                            reward = self.food_reward * (0.5 ** manhattan_dist)
                            self.rewards[x][y] += reward

    def value_iteration(self):
        """
        Implements value iteration to compute utilities for each state.
        Continues iterating until convergence threshold is reached or max iterations reached.
        """
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
            
            # Reached convergence
            if max_change < self.convergence_threshold:
                break

    def compute_state_utility(self, x, y):
        """
        Calculates the utility of a given state (x,y).
        - Gets the immediate reward R for the state
        - Checks if the state is in a danger zone
        - Applies discount factor to the maximum expected utility of the state based on danger zone
        - Finds the maximum expected utility of the state based on all possible actions

        Parameters:
        x (int): x-coordinate of the state
        y (int): y-coordinate of the state

        Returns:
        float: The resulting utility of the state
        """
        # No reward for invalid states
        if self.rewards[x][y] is None:
            return None
            
        # Get the immediate reward for the current state
        R = self.rewards[x][y]
        
        in_danger_zone = False
        # Get current positions and states of all ghosts
        ghost_states = api.ghostStates(self.current_state)

        # Check each ghost to see if agent is in its danger radius
        for (ghost_x, ghost_y), scared in ghost_states:
            # Only consider non-scared ghosts as threats
            if not scared:
                # Convert ghost coordinates to integers
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                # Calculate Manhattan distance to ghost
                manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
                # If within danger radius, mark as dangerous and stop checking
                if manhattan_dist <= self.danger_radius:
                    in_danger_zone = True
                    break
        
        # Use lower discount factor if in danger zone to encourage immediate rewards
        # Otherwise use normal discount factor for long-term planning
        discount = self.danger_discount if in_danger_zone else self.discount
        
        # Find the maximum expected utility across all possible actions
        max_utility = float("-inf")
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Calculate expected utility for this action considering movement uncertainty
            exp_utility = self.get_expected_utility(x, y, action)
            max_utility = max(max_utility, exp_utility)
            
        # Return Bellman equation
        return R + discount * max_utility

    def get_expected_utility(self, x, y, action):
        """
        Calculates the expected utility of taking an action in a given state (x,y).
        For each possible next state:
        - Gets the transition probability
        - Multiplies the probability by the utility of the next state
        - Sums up the weighted utilities to get the expected utility
        The boundary cases are handled by assuming the agent stays in the same state.

        Parameters:
        x (int): x-coordinate of the state
        y (int): y-coordinate of the state
        action (int): The action to take (e.g., Directions.NORTH)

        Returns:
        float: The expected utility of taking the given action in the state
        """
        # Get dictionary of possible next states and their probabilities
        # for the given action (e.g., moving NORTH might have 0.8 prob forward, 0.1 left, 0.1 right)
        successors = self.get_successor_states(x, y, action)
        
        # Initialize expected utility to 0
        exp_utility = 0.0
        # Iterate through each possible next state and its probability
        for (next_x, next_y), prob in successors.items():
            # Check if next state is valid (within bounds and not a wall)
            if (0 <= next_x < self.width and 
                0 <= next_y < self.height and 
                self.utilities[next_x][next_y] is not None):
                # Add probability-weighted utility of next state
                exp_utility += prob * self.utilities[next_x][next_y]
            else:
                # If next state is invalid (wall/out of bounds),
                # use current state's utility instead (agent stays put)
                exp_utility += prob * self.utilities[x][y]
            
        # Return total expected utility for this action
        return exp_utility

    def get_successor_states(self, x, y, action):
        """
        Gets the possible next states and their probabilities given an action.
        This is used for modelling the agent's movement uncertainty.

        Parameters:
        x (int): x-coordinate the state
        y (int): y-coordinate the state

        Returns:
        dict: Dictionary of possible next states and their probabilities
        """
        # Dictionary to store possible next states and their probabilities
        successors = {}
        
        # Define movement vectors based on intended action
        # Each action has:
        # - intended direction (80% probability)
        # - two perpendicular directions (10% probability each)
        if action == Directions.NORTH:
            intended = (0, 1)      # Move up
            perpendicular = [(1, 0), (-1, 0)]  # Right and left
        elif action == Directions.SOUTH:
            intended = (0, -1)     # Move down
            perpendicular = [(1, 0), (-1, 0)]  # Right and left
        elif action == Directions.EAST:
            intended = (1, 0)      # Move right
            perpendicular = [(0, 1), (0, -1)]  # Up and down
        elif action == Directions.WEST:
            intended = (-1, 0)     # Move left
            perpendicular = [(0, 1), (0, -1)]  # Up and down
            
        # Calculate intended next position (0.8 probability)
        next_x = x + intended[0]
        next_y = y + intended[1]
        successors[(next_x, next_y)] = 0.8
        
        # Calculate perpendicular positions (0.1 probability each)
        for dx, dy in perpendicular:
            next_x = x + dx
            next_y = y + dy
            successors[(next_x, next_y)] = 0.1
            
        return successors

    def getAction(self, state):
        """
        Responsible for making the decisions.
        - Updates the internal state representation
        - Gets the current position of the agent
        - Gets the legal actions available (by calculating expected utilities for each action and choosing the best one) and removes the STOP action
        - Chooses the best action based on the expected utility of each action

        Parameters:
        state (GameState): The current game state

        Returns:
        str: The chosen action (e.g., Directions.NORTH)
        """
        self.update_state(state)
        
        x, y = api.whereAmI(state)
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
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
        """Reset all instance variables between games"""
        self.current_state = None
        self.width = 0
        self.height = 0
        self.utilities = None
        self.rewards = None
        self.danger_radius = 0
        self.food_radius = 0