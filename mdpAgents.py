# -*- coding: utf-8 -*-

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):
    """A Pacman agent that uses Modified Policy Iteration (MPI) to make decisions.
    
    This agent implements a Markov Decision Process (MDP) solution where:
    - States track Pacman position, ghost positions, and wall locations
    - Actions are the legal moves (North, South, East, West)
    - Rewards include penalties for ghosts and wall collisions
    - Transitions have uncertainty (0.8 intended, 0.1 perpendicular directions)
    
    The agent uses Modified Policy Iteration to find an optimal policy by alternating between:
    1. Limited policy evaluation: V(s) = R(s) + gamma * Σ P(s'|s,pi(s)) * V(s')
    2. Policy improvement: pi(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + gamma*V(s')]
    """

    def __init__(self):
        # MDP PARAMETERS
        # Discount factor gamma: Determines how much future rewards are valued compared to immediate ones
        self.discount = 0.9 
        # Controls number of value function updates before policy improvement
        self.k = 3  
        # Convergence threshold: Defines when value iteration should stop based on value changes
        self.theta = 0.01  
        
        # MOVEMENT PARAMETERS
        # Probability that Pacman moves in the direction it intends to
        self.prob_intended = 0.8  
        # Probability of deviating 90 degrees from intended direction
        self.prob_perpendicular = 0.1  
        
        # GHOST PARAMETERS
        # Base value for how much Pacman is penalized for being near ghosts
        self.ghost_penalty = -100  
        # Maximum distance at which ghosts are considered threatening
        self.ghost_distance_threshold = 4  
        # Factor that controls how ghost penalty scales with distance
        self.ghost_penalty_divisor = 1  
        
        # COLLISION PARAMETERS
        # Penalty applied when Pacman attempts to move into a wall
        self.wall_collision_penalty = -10  
        
        # STATE TRACKING
        # Stores the value function mapping states to their estimated values
        self.V = util.Counter()  
        # Stores the policy mapping states to optimal actions
        self.pi = util.Counter()  

    def registerInitialState(self, state):
        """Reset the agent's value function and policy for a new game.
        
        Initializes empty Counters for both V (state values) and π (policy) at game start.
        """
        # Reset values for new game
        self.V = util.Counter()
        self.pi = util.Counter()
        
    def getAction(self, state):
        """Determine the next action for Pacman using the MDP policy.
        
        Runs modified policy iteration to update values and policy, then returns either:
        - The best action according to current policy if legal
        - A random legal action if best action is not legal
        """
        # Get legal actions
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Get current state key
        current_state = self.getStateKey(state)
        
        # Run modified policy iteration
        self.modifiedPolicyIteration(state, legal)
        
        # Get best action according to policy
        best_action = self.pi[current_state]
        
        if best_action in legal:
            return api.makeMove(best_action, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
            
    def getStateKey(self, state):
        """Create a hashable representation of the game state.
        
        Converts the state into a tuple containing:
        - Pacman's position (x,y)
        - Ghost positions as sorted tuple
        - Wall positions as sorted tuple
        """
        pacman_pos = api.whereAmI(state)
        ghost_positions = tuple(sorted(api.ghosts(state)))
        walls = tuple(sorted(api.walls(state)))
        return (pacman_pos, ghost_positions, walls)

    def getNextState(self, state_key, action):
        """Predict the next state given current state and action.
        
        Computes next position based on action direction and checks for wall collisions.
        Returns new state key with updated Pacman position.
        """
        pacman_pos, ghost_positions, walls = state_key
        x, y = pacman_pos
        
        # Get next position based on action
        if action == Directions.NORTH:
            next_pos = (x, y + 1)
        elif action == Directions.SOUTH:
            next_pos = (x, y - 1)
        elif action == Directions.EAST:
            next_pos = (x + 1, y)
        elif action == Directions.WEST:
            next_pos = (x - 1, y)
        
        # Check if next position is a wall
        if next_pos in walls:
            next_pos = pacman_pos
            
        return (next_pos, ghost_positions, walls)
            
    def getReward(self, state_key, action, next_state_key):
        """Calculate the immediate reward for a state transition.
        
        Rewards are determined by:
        - Wall collision penalty (-10)
        - Ghost proximity penalty: ghost_penalty / (distance + 1)
        where penalties increase as distance to ghost decreases
        """
        current_pos = state_key[0]
        ghost_positions = state_key[1]
        next_pos = next_state_key[0]
        
        # If position didn't change (hit wall), penalize
        if current_pos == next_pos:
            return self.wall_collision_penalty
        
        reward = 0
        # Check ghost distances from next position
        for ghost_pos in ghost_positions:
            dist = util.manhattanDistance(next_pos, ghost_pos)
            if dist < self.ghost_distance_threshold:
                reward += self.ghost_penalty / (dist + self.ghost_penalty_divisor)
                    
        return reward
        
    def getTransitionProb(self, action):
        """Get probability distribution over actual movements for intended action.
        
        Movement uncertainty model:
        - P(intended direction) = 0.8
        - P(perpendicular directions) = 0.1 each
        Returns Counter mapping directions to probabilities.
        """
        probs = util.Counter()
        
        if action == Directions.NORTH:
            probs[Directions.NORTH] = self.prob_intended
            probs[Directions.EAST] = self.prob_perpendicular
            probs[Directions.WEST] = self.prob_perpendicular
        elif action == Directions.SOUTH:
            probs[Directions.SOUTH] = self.prob_intended
            probs[Directions.EAST] = self.prob_perpendicular
            probs[Directions.WEST] = self.prob_perpendicular
        elif action == Directions.EAST:
            probs[Directions.EAST] = self.prob_intended
            probs[Directions.NORTH] = self.prob_perpendicular
            probs[Directions.SOUTH] = self.prob_perpendicular
        elif action == Directions.WEST:
            probs[Directions.WEST] = self.prob_intended
            probs[Directions.NORTH] = self.prob_perpendicular
            probs[Directions.SOUTH] = self.prob_perpendicular
            
        return probs
        
    def modifiedPolicyIteration(self, state, legal_actions):
        """Update value function and policy using Modified Policy Iteration (MPI).
        
        Performs k steps of policy evaluation using Bellman equation:
        V(s) = R(s) + γ * Σ P(s'|s,π(s)) * V(s')
        
        Followed by policy improvement:
        π(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]
        
        Stops evaluation early if value change < theta threshold.
        """
        current_state_key = self.getStateKey(state)
        
        # Policy evaluation for k steps
        for _ in range(self.k):
            delta = 0
            v = self.V[current_state_key]
            
            # If no policy for state yet, initialize randomly
            if current_state_key not in self.pi:
                self.pi[current_state_key] = random.choice(legal_actions)
            
            action = self.pi[current_state_key]
            new_v = 0
            
            # Get transition probabilities for intended action
            trans_probs = self.getTransitionProb(action)
            
            # Calculate expected value over all possible outcomes
            for actual_action, prob in trans_probs.items():
                # Get next state for this actual action
                next_state_key = self.getNextState(current_state_key, actual_action)
                # Get reward for this transition
                reward = self.getReward(current_state_key, actual_action, next_state_key)
                # Update value using Bellman equation
                new_v += prob * (reward + self.discount * self.V[next_state_key])
            
            self.V[current_state_key] = new_v
            delta = abs(v - new_v)
            
            if delta < self.theta:
                break
                
        # Policy improvement
        old_action = self.pi[current_state_key]
        
        # Find best action
        max_value = float('-inf')
        best_action = None
        
        # Try all legal actions
        for action in legal_actions:
            value = 0
            trans_probs = self.getTransitionProb(action)
            
            # Calculate expected value over all possible outcomes
            for actual_action, prob in trans_probs.items():
                next_state_key = self.getNextState(current_state_key, actual_action)
                reward = self.getReward(current_state_key, actual_action, next_state_key)
                value += prob * (reward + self.discount * self.V[next_state_key])
                
            if value > max_value:
                max_value = value
                best_action = action
                
        self.pi[current_state_key] = best_action
        
    def final(self, state):
        """Reset agent's value function and policy when game ends.
        
        Clears V and π Counters to prepare for next game.
        """
        self.V = util.Counter()
        self.pi = util.Counter()