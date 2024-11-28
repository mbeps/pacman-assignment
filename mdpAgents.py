from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):
    """
    A Pacman agent implementing Markov Decision Process (MDP) with Value Iteration.
    
    This agent enhances the standard MDP/Value Iteration framework through:
    
    1. Sophisticated Reward Structure:
    - Progressive reward propagation that respects maze topology
    - Separate decay rates for food rewards and ghost threats
    - Dynamic adjustment of rewards based on ghost states (scared/normal)
    - Spawn point penalties to encourage strategic movement
    
    2. Advanced MDP Components:
    - State Space: Complete maze grid including walls and boundaries
    - Actions: Cardinal directions (North, South, East, West)
    - Transitions: Stochastic movement (0.8 intended, 0.1 perpendicular)
    - Rewards: Complex reward landscape from multiple sources
    - Value Function: Iteratively computed utilities
    
    3. Enhanced Value Iteration:
    - Configurable convergence threshold and iteration limit
    - Efficient utility updates using temporary grid
    - Balanced exploration through discount factor
    - Strategic depth through living reward adjustment
    
    4. Intelligent Reward Propagation:
    - Distance-aware reward decay through valid paths
    - Separate influence radii for threats and goals
    - Exponential decay factors for natural gradients
    - Topology-respecting reward spread
    
    Core Parameters:
    - MDP: discount, living_reward
    - Threats: danger_discount, ghost_reward, danger_radius, danger_decay
    - Goals: food_reward, food_radius, food_decay
    - Algorithm: iterations, convergence_threshold
    - Special States: scared_ghost_reward, spawn_penalty
    
    This implementation maintains MDP principles while introducing sophisticated
    reward engineering and value computation strategies that improve agent behaviour
    through more nuanced state evaluation.
    """
    def __init__(self):
        # --- Grid State Parameters ---
        # Represents the game board layout
        self.grid = None
        # Stores utility values for each cell
        self.utilities = None  
        # Stores reward values for each cell
        self.rewards = None
        # Grid dimensions
        self.width = None 
        self.height = None
        
        # --- Core MDP Parameters ---
        # Discount factor (0-1): Lower values make agent more short-sighted
        self.discount = None
        # Reward/penalty for each move: More negative makes agent move faster
        self.living_reward = None
        
        # --- Ghost/Danger Parameters ---
        # How much to reduce utilities near ghosts: Higher values increase ghost avoidance
        self.danger_discount = None
        # Base penalty for ghost cells: More negative means stronger ghost avoidance 
        self.ghost_reward = None  
        # How far ghost effects spread: Larger means wider avoidance area
        self.danger_radius = None
        # How quickly ghost penalty decreases with distance: Higher means faster dropoff
        self.danger_decay = None
        
        # --- Food Parameters ---
        # Base reward for food cells: Higher values increase food seeking
        self.food_reward = None
        # How far food effects spread: Larger means earlier food seeking
        self.food_radius = None
        # How quickly food reward decreases with distance: Higher means faster dropoff
        self.food_decay = None
        
        # --- Algorithm Parameters ---
        # Number of value iteration updates: More iterations = more accurate but slower
        self.iterations = 1500
        # Minimum change to continue iterating: Lower = more precise but slower
        self.convergence_threshold = 0.001
    
        # --- Special Ghost States ---
        # Reward for eating scared ghost: Higher encourages ghost hunting
        self.scared_ghost_reward = 200  
        # Distance considered safe from scared ghost: Higher means more caution
        self.scared_safety_threshold = 3  
    
        # --- Spawn Point Parameters ---
        # List of ghost spawn locations
        self.ghost_spawn_points = []
        # Penalty near spawn points: More negative means stronger spawn avoidance
        self.spawn_penalty = -50
        # How far spawn effects spread: Larger means wider avoidance area
        self.spawn_radius = 2
        # How quickly spawn penalty decreases: Higher means faster dropoff
        self.spawn_decay = 0.6
    
    def registerInitialState(self, state):
        """
        Initialises the MDP agent when a new game starts.
        It sets up:
        - The grid representation of the game
        - The dimensions of the grid
        - The danger and food radii based on the number of ghosts
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
            self.discount = 0.8  
            self.living_reward = -0.04
            self.danger_discount = 0.95
            self.ghost_reward = -1000
            self.danger_radius = 4
            self.danger_decay = 0.7
            self.food_reward = 30 
            self.food_radius = 3
            self.food_decay = 0.5
        elif ghost_count == 1:  
            self.discount = 0.85
            self.living_reward = -0.1
            self.danger_discount = 0.9
            self.ghost_reward = -75
            self.danger_radius = 3
            self.danger_decay = 0.8
            self.food_reward = 75
            self.food_radius = 3
            self.food_decay = 0.7
        
        self.utilities = self.create_grid(0.0)
        self.rewards = self.create_grid(self.living_reward)
        
        walls = api.walls(state)
        for x, y in walls:
            self.rewards[x][y] = None
            self.utilities[x][y] = None
            
        self.update_state(state)
    
        # Store ghost spawn points from initial positions
        self.ghost_spawn_points = [(int(x), int(y)) for x, y in api.ghosts(state)]

    def create_grid(self, initial_value):
        """
        Creates a 2D grid with the given initial value.
    
        Parameters:
        initial_value (float): The initial value to fill the grid with
    
        Returns:
        list: A 2D list representing the initialised grid
        """
        return [[initial_value for y in range(self.height)] 
                for x in range(self.width)]

    def update_state(self, state):
        """
        Continuously updates the internal state representation of the agent.
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
        Updates the rewards grid using progressive propagation that respects walls.
    
        This method enhances the standard MDP reward structure by implementing a more 
        sophisticated reward propagation system. While traditional MDPs often use simple
        distance-based rewards, this implementation creates a more realistic reward 
        landscape by:
    
        1. Progressive Propagation:
        - Instead of using Manhattan distance, rewards spread step-by-step through
            connected cells
        - Each step applies the appropriate decay factor (danger_decay for ghosts,
            food_decay for food)
        - This creates a natural gradient that follows the actual maze pathways
    
        2. Relationship to MDP Framework:
        - The rewards computed here serve as the 'R(s)' component in the Bellman equation
        - While Value Iteration computes long-term utilities, these rewards represent
            immediate payoffs/penalties for being in each state
        - The decay factors (0.7 for danger, 0.5 for food) create exponential falloff
            within their respective radii (danger_radius and food_radius)
        
        3. Decay Factor Behaviour:
        - Ghost penalties: Uses danger_decay (0.7) for slower falloff, creating wider
            danger zones that encourage early ghost avoidance
        - Food rewards: Uses food_decay (0.5) for faster falloff, creating focused
            attractors that encourage direct paths to food
        - Within radius: reward * (decay_factor ^ steps_from_source)
        - Outside radius: only base living_reward applies
        
        This enhancement improves the MDP by:
        - Creating more accurate immediate rewards that respect maze topology
        - Maintaining exponential decay while ensuring rewards only propagate through
        valid paths
        - Allowing different decay rates for threats vs rewards, enabling more nuanced
        behaviour
        
        Parameters:
        state (GameState): The current game state containing ghost and food positions
        """
        self.current_state = state
        
        # Reset non-wall states to living reward
        for x in range(self.width):
            for y in range(self.height):
                if self.rewards[x][y] is not None:
                    self.rewards[x][y] = self.living_reward
    
        # Process ghost states first (stronger influence)
        ghost_states = api.ghostStates(state)
        for (ghost_x, ghost_y), scared_timer in ghost_states:
            ghost_x, ghost_y = int(ghost_x), int(ghost_y)
            
            # Initialise queue differently based on ghost state
            if scared_timer <= self.scared_safety_threshold:
                # Treat as dangerous ghost
                queue = [(ghost_x, ghost_y, self.ghost_reward, 0)]
            else:
                # Weight reward based on remaining time
                time_weight = min(1.0, scared_timer / 20.0)
                weighted_reward = self.scared_ghost_reward * time_weight
                queue = [(ghost_x, ghost_y, weighted_reward, 0)]
            
            visited = set()
            
            while queue:
                x, y, value, dist = queue.pop(0)
                if dist > self.danger_radius:
                    continue
                
                if self.rewards[x][y] is not None:
                    if scared_timer <= self.scared_safety_threshold:
                        # For dangerous ghosts, take minimum (most negative)
                        self.rewards[x][y] = min(self.rewards[x][y], value)
                    else:
                        # For scared ghosts, add reward
                        self.rewards[x][y] += value
                
                # Add valid neighbours with decayed values
                for next_x, next_y in self.get_valid_neighbors(x, y):
                    if (next_x, next_y) not in visited:
                        next_value = value * self.danger_decay
                        next_dist = dist + 1
                        queue.append((next_x, next_y, next_value, next_dist))
                        visited.add((next_x, next_y))
    
            if scared_timer > self.scared_safety_threshold:
                # Process spawn points when ghosts are scared
                for spawn_x, spawn_y in self.ghost_spawn_points:
                    # Initialise queue with spawn point
                    spawn_queue = [(spawn_x, spawn_y, self.spawn_penalty, 0)]
                    spawn_visited = set()
                    
                    while spawn_queue:
                        x, y, penalty, dist = spawn_queue.pop(0)
                        if dist > self.spawn_radius:
                            continue
                            
                        if self.rewards[x][y] is not None:
                            # Add penalty to current cell
                            self.rewards[x][y] += penalty
                            
                        # Add valid neighbours with decayed penalty
                        for next_x, next_y in self.get_valid_neighbors(x, y):
                            if (next_x, next_y) not in spawn_visited:
                                next_penalty = penalty * self.spawn_decay
                                next_dist = dist + 1
                                spawn_queue.append((next_x, next_y, next_penalty, next_dist))
                                spawn_visited.add((next_x, next_y))
    
        # Process food rewards with propagation
        food_locations = api.food(state)
        for food_x, food_y in food_locations:
            self.rewards[food_x][food_y] += self.food_reward
            
            # Initialise queue with food position
            queue = [(food_x, food_y, self.food_reward, 0)]
            visited = set()
            
            while queue:
                x, y, reward, dist = queue.pop(0)
                if dist > self.food_radius:
                    continue
                
                # Add reward (allows accumulation of multiple food influences)
                if self.rewards[x][y] is not None and (x, y) != (food_x, food_y):
                    self.rewards[x][y] += reward
                
                # Add valid neighbours to queue
                for next_x, next_y in self.get_valid_neighbors(x, y):
                    if (next_x, next_y) not in visited:
                        next_reward = reward * self.food_decay
                        next_dist = dist + 1
                        queue.append((next_x, next_y, next_reward, next_dist))
                        visited.add((next_x, next_y))

    def get_valid_neighbors(self, x, y):
        """
        Gets valid neighbouring cells that aren't walls, used for reward propagation.
    
        This method supports the progressive reward propagation by identifying legal adjacent 
        cells. The definition of a valid neighbour is:
        1. Must be directly adjacent (no diagonals)
        2. Must be within grid boundaries
        3. Must not be a wall (rewards[x][y] is not None)
        
        This ensures rewards only propagate through paths that Pacman could actually traverse,
        making the reward landscape more accurately reflect the maze structure.
        
        Parameters:
        x (int): x-coordinate of the current cell
        y (int): y-coordinate of the current cell
        
        Returns:
        list: List of (x,y) tuples representing valid adjacent cells
        """
        neighbors = []
        # Check each adjacent cell (no diagonals)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            # Check if neighbour is within bounds and not a wall
            if (0 <= next_x < self.width and 
                0 <= next_y < self.height and 
                self.rewards[next_x][next_y] is not None):
                neighbors.append((next_x, next_y))
        return neighbors

    def value_iteration(self):
        """
        Implements value iteration to compute optimal state utilities in the MDP framework.
        
        This method implements the core Value Iteration algorithm for Markov Decision 
        Processes (MDPs) by:
        
        1. Iterative Computation:
        - Repeatedly updates utility estimates for each state until convergence
        - Uses the Bellman equation to compute expected future rewards
        - Maintains a convergence threshold to determine when to stop
        
        2. MDP Components:
        - State space: Grid cells representing valid Pacman positions
        - Actions: North, South, East, West movements
        - Transitions: Probabilistic movement outcomes
        - Rewards: Immediate payoffs from the reward grid
        - Discount factor: Weighs future vs immediate rewards
        
        3. Convergence Behaviour:
        - Continues until change in utilities falls below convergence_threshold
        - Maximum iterations prevent infinite loops
        - Updates utilities in place using a temporary grid
        - Tracks maximum change per iteration for convergence check
        
        This implementation enhances the MDP solution by:
        - Ensuring systematic utility propagation through the state space
        - Balancing computational efficiency with solution accuracy
        - Providing guaranteed convergence to optimal utilities
        
        The computed utilities then drive optimal policy selection by:
        - Enabling evaluation of expected outcomes for each action
        - Supporting selection of actions that maximise expected utility
        - Accounting for both immediate rewards and future consequences
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
        Calculates the utility of a specific state position (x,y) in the MDP framework.
        
        This method implements the state utility computation component of Value Iteration by:
        
        1. State Evaluation:
        - Retrieves the immediate reward R(s) for the current state
        - Considers danger zones from ghost positions
        - Applies the discount factor γ to future rewards
        - Computes maximum expected utility across all actions
        
        2. MDP Components Used:
        - Transition model: P(s'|s,a) for movement uncertainty
        - Reward function: R(s) from the reward grid
        - Discount factor: γ for future reward weighting
        - Action set: Available movements from current position
        
        3. Utility Calculation:
        - Implements U(s) = R(s) + γ * max_a Σ P(s'|s,a)U(s')
        - Handles wall collisions and boundary conditions
        - Incorporates movement uncertainty in expected outcomes
        - Returns the optimal utility value for the state
        
        Parameters:
        x (int): The x-coordinate of the state position
        y (int): The y-coordinate of the state position
        
        Returns:
        float: The computed utility value for state (x,y)
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
                ghost_x, ghost_y = int(ghost_x), int(ghost_y)
                manhattan_dist = abs(x - ghost_x) + abs(y - ghost_y)
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
            
        return R + discount * max_utility

    def get_expected_utility(self, x, y, action):
        """
        Calculates the expected utility of taking an action in a given state (x,y).
        
        This method implements the expected utility computation for MDP value iteration by:
        
        1. Action Evaluation:
        - Computes sum(P(s'|s,a)U(s')) for a specific action
        - Handles movement uncertainty using the transition model
        - Accounts for boundary conditions and wall collisions
        - Incorporates the stored utilities of neighbouring states
        
        2. MDP Components Used:
        - Transition probabilities: P(s'|s,a) for intended and unintended movements
        - Current utility values: U(s') from the utilities grid
        - Action effects: Directional movement outcomes
        - State adjacency: Valid neighbouring positions
        
        3. Probability Calculations:
        - Intended direction: 0.8 probability of success
        - Perpendicular directions: 0.1 probability each
        - Collision cases: Probability redistributed to current state
        - Returns weighted sum of possible outcome utilities
        
        Parameters:
        x (int): x-coordinate of the current state
        y (int): y-coordinate of the current state
        action (int): The action to evaluate (from Directions enumeration)
        
        Returns:
        float: The expected utility value for the action in state (x,y)
        """
        # Get dictionary of possible next states and their probabilities for the given action (e.g., moving NORTH might have 0.8 prob forward, 0.1 left, 0.1 right)
        successors = self.get_successor_states(x, y, action)
        
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
        
        This method implements the transition model for the MDP framework by:
        
        1. Movement Model:
        - Determines all possible next states from current position
        - Handles stochastic movement with directional uncertainty
        - Accounts for wall collisions and boundary conditions
        - Maps state transitions to their respective probabilities
        
        2. MDP Components Used:
        - State space: Valid grid positions
        - Actions: Directional movements (North, South, East, West)
        - Transition function: P(s'|s,a) probability distribution
        - Environment constraints: Wall positions and grid boundaries
        
        3. Probability Distribution:
        - Main direction: 0.8 probability of intended movement
        - Side directions: 0.1 probability for each perpendicular movement
        - Collision handling: Probabilities redirect to current state
        - Returns complete mapping of next states to probabilities
        
        Parameters:
        x (int): x-coordinate of the current state
        y (int): y-coordinate of the current state
        action (int): The action being evaluated
        
        Returns:
        dict: Mapping of possible next states to their transition probabilities
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
        Determines optimal action selection using MDP value iteration results.
        
        This method implements the policy execution component of the MDP framework by:
        
        1. State Processing:
        - Updates internal MDP state representation
        - Processes current Pacman position coordinates
        - Identifies available legal actions in current state
        - Removes STOP from action consideration
        
        2. MDP Policy Implementation:
        - Uses converged utility values from value iteration
        - Evaluates expected utility for each legal action
        - Handles probabilistic transition outcomes
        - Selects action maximising expected future reward
        
        3. Decision Making:
        - Computes action utilities using current value function
        - Considers movement uncertainty in expectations
        - Applies optimal policy from value iteration
        - Returns best action according to MDP solution
        
        Parameters:
        state (GameState): The current game state containing agent positions
        
        Returns:
        str: The optimal action from Directions enumeration
        """
        self.update_state(state)
        #! For debugging 
        # self.print_grid(self.utilities)
        
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

    def print_grid(self, grid):
        """Prints a formatted view of the utilities grid with coloured values"""
        # ANSI colour codes
        GREEN = '\033[92m'      # Positive values
        RED = '\033[91m'        # Negative values
        BLUE = '\033[94m'       # Zero values
        RESET = '\033[0m'
        PACMAN = '\033[43m'     # Yellow background for Pacman
        GHOST = '\033[101m'     # Bright red background for ghosts
        BOLD = '\033[1m'        # Bold text
        
        # Get positions
        pacman_pos = api.whereAmI(self.current_state)
        ghost_positions = api.ghosts(self.current_state)
        
        for y in range(self.height-1, -1, -1):
            row = ""
            for x in range(self.width):
                if grid[x][y] is None:  # Wall
                    row += "          "     # 10 spaces for walls
                else:
                    value = grid[x][y]
                    if value > 0:
                        colour = GREEN
                    elif value < 0:
                        colour = RED
                    else:
                        colour = BLUE
                    
                    # Add highlights for Pacman and ghosts
                    if (x, y) == pacman_pos:
                        row += PACMAN + BOLD + colour + "{:8.2f}  ".format(value) + RESET
                    elif (x, y) in ghost_positions:
                        row += GHOST + BOLD + colour + "{:8.2f}  ".format(value) + RESET
                    else:
                        row += colour + "{:8.2f}  ".format(value) + RESET
            print(row)
        print("-" * (self.width * 10))

    def final(self, state):
        """Reset all instance variables between games"""
        self.current_state = None
        self.width = 0
        self.height = 0
        self.utilities = None
        self.rewards = None
        self.danger_radius = 0
        self.food_radius = 0