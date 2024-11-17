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
        self.discount = 0.7
        self.danger_discount = 0.95
        self.living_reward = -0.04
        self.food_reward = 10
        self.ghost_reward = -1000
        self.danger_radius = 4
        self.food_radius = 3  # Smaller radius for food propagation
        self.iterations = 100
        self.convergence_threshold = 0.01

    def registerInitialState(self, state):
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
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def update_state(self, state):
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward
        
        self.update_rewards(state)
        self.value_iteration()

    def update_rewards(self, state):
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
        if self.rewards[x][y] is None:
            return None
            
        R = self.rewards[x][y]
        
        in_danger_zone = False
        ghost_states = api.ghostStates(self.current_state)
        for (ghost_x, ghost_y), scared in ghost_states:
            if not scared:
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
                if manhattan_dist <= self.danger_radius:
                    in_danger_zone = True
                    break
        
        discount = self.danger_discount if in_danger_zone else self.discount
        
        max_utility = float("-inf")
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            exp_utility = self.get_expected_utility(x, y, action)
            max_utility = max(max_utility, exp_utility)
            
        return R + discount * max_utility

    def get_expected_utility(self, x, y, action):
        successors = self.get_successor_states(x, y, action)
        
        exp_utility = 0.0
        for (next_x, next_y), prob in successors.items():
            if (0 <= next_x < self.width and 
                0 <= next_y < self.height and 
                self.utilities[next_x][next_y] is not None):
                exp_utility += prob * self.utilities[next_x][next_y]
            else:
                exp_utility += prob * self.utilities[x][y]
            
        return exp_utility

    def get_successor_states(self, x, y, action):
        successors = {}
        
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
            
        next_x = x + intended[0]
        next_y = y + intended[1]
        successors[(next_x, next_y)] = 0.8
        
        for dx, dy in perpendicular:
            next_x = x + dx
            next_y = y + dy
            successors[(next_x, next_y)] = 0.1
            
        return successors

    def getAction(self, state):
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
        self.current_state = None