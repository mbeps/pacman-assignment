from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):
    def __init__(self):
        # MDP Parameters
        self.discount = 0.9  # Discount factor gamma
        self.k = 3  # Number of value iterations per policy iteration
        self.theta = 0.01  # Convergence threshold
        
        # Reward Parameters
        self.ghost_penalty = -100  # Penalty for being near a ghost
        self.ghost_distance_threshold = 3  # Distance to consider ghost threat
        
        # Movement probability parameters (from api.py)
        self.prob_intended = 0.8  # Probability of moving in intended direction
        self.prob_perpendicular = 0.1  # Probability of moving perpendicular
        
        # State tracking
        self.V = util.Counter()  # Value function
        self.pi = util.Counter()  # Policy

    def registerInitialState(self, state):
        # Reset values for new game
        self.V = util.Counter()
        self.pi = util.Counter()
        
    def getAction(self, state):
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
        """Convert state to hashable key - only tracking Pacman and ghost positions"""
        pacman_pos = api.whereAmI(state)
        ghost_positions = tuple(sorted(api.ghosts(state)))
        walls = tuple(sorted(api.walls(state)))
        return (pacman_pos, ghost_positions, walls)

    def getNextState(self, state_key, action):
        """Predict next state given current state and action"""
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
        """Calculate reward for state-action-nextstate transition"""
        current_pos = state_key[0]
        ghost_positions = state_key[1]
        next_pos = next_state_key[0]
        
        # If position didn't change (hit wall), penalize
        if current_pos == next_pos:
            return -10
        
        reward = 0
        # Check ghost distances from next position
        for ghost_pos in ghost_positions:
            dist = util.manhattanDistance(next_pos, ghost_pos)
            if dist < self.ghost_distance_threshold:
                reward += self.ghost_penalty / (dist + 1)
                    
        return reward
        
    def getTransitionProb(self, action):
        """Get transition probabilities for action"""
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
        """Modified Policy Iteration implementation"""
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
        """Reset state for new game"""
        self.V = util.Counter()
        self.pi = util.Counter()